import 'dart:io';
import 'dart:isolate';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart' hide Image;
import 'package:flutter_face_sdk/converter.dart' show ImageConverter;
import 'package:flutter_face_sdk/flutter_face_sdk.dart'
    show
        BufferInfo,
        Error,
        FacePosition,
        FaceNotFoundError,
        FacialFeatures,
        Image,
        Int64Buffer,
        Tracker,
        GetValueConfidence;
import 'package:flutter_face_sdk/flutter_face_sdk.dart' as FSDK;
import 'package:path_provider/path_provider.dart';

class _FacesTrackerIsolateInfo {
  final SendPort port;
  final int trackerHandle;
  final BufferInfo idsBufferInfo;

  _FacesTrackerIsolateInfo(this.port, this.trackerHandle, this.idsBufferInfo);
}

class _WorkerData {
  final int image;
  final int orientation;
  final bool frontFacing;

  _WorkerData(this.image, this.orientation, this.frontFacing);
}

class FaceWrapper {
  final int _id;
  final Tracker _tracker;

  String? _name;
  FacePosition? _position;
  FacialFeatures? _features;

  double? _liveness;

  FaceWrapper(this._id, this._tracker);

  int get id => _id;

  String get name {
    if (_name == null) {
      _tracker.lockID(_id);
      _name = _tracker.getAllNames(_id);
      _tracker.unlockID(_id);
    }

    return _name!;
  }

  FacePosition get position {
    _position ??= _tracker.getFacePosition(0, _id);
    return _position!;
  }

  FacialFeatures get features {
    _features ??= _tracker.getFacialFeatures(0, _id);
    return _features!;
  }

  double get liveness {
    return _liveness!;
  }

  bool checkLiveness() {
    _tracker.lockID(_id);
    try {
      String livenessAttribute =
          _tracker.getFacialAttribute(0, _id, "Liveness");
      _liveness = GetValueConfidence(livenessAttribute, "Liveness");
    } on FSDK.AttributeNotDetectedError {
      return false;
    } finally {
      _tracker.unlockID(_id);
    }
    return true;
  }
}

enum FaceTrackerState {
  notInitialized,
  initializing,
  waitingForImage,
  waitingForIds,
  idsReady
}

class FaceMatchResult {
  final String name;
  final int id;
  final double similarity;
  final double liveness;
  final Image? image;

  FaceMatchResult(
    this.name,
    this.id,
    this.similarity,
    this.liveness, {
    this.image,
  });
}

class FacesTracker extends ChangeNotifier {
  static const _path = 'tracker.bin';

  String _trackerPath = "";
  FaceTrackerState _state = FaceTrackerState.notInitialized;
  FaceTrackerState get state => _state;

  void _onStateChange(FaceTrackerState newState) {
    bool hasChanged = _state != newState;
    _state = newState;
    if (hasChanged) {
      notifyListeners();
    }
  }

  late SendPort _send;
  late Isolate _isolate;
  Image? _fsdkImage;

  late final _tracker = Tracker();
  final _receive = ReceivePort();
  final _converter = ImageConverter();
  final _ids = Int64Buffer.allocate(5);

  int get width => _converter.width;
  int get height => _converter.height;

  final double similarityThreshold;

  FacesTracker({
    required this.similarityThreshold,
  });

  void saveTracker() {
    _tracker.saveToFile(_trackerPath);
  }

  @override
  void dispose() {
    _isolate.kill(priority: Isolate.immediate);

    _converter.free();

    saveTracker();
    _tracker.free();

    super.dispose();
  }

  Future<void> _openTracker() async {
    final directory = await getApplicationDocumentsDirectory();
    _trackerPath = '${directory.path}/$_path';

    try {
      Tracker.fromFile(_trackerPath, tracker: _tracker);
    } on Error {
      // Couldn't load tracker from memory, file may not exist
    }

    _setTrackerParameters();
  }

  void _setTrackerParameters() {
    _tracker.setMultipleParameters({
      'ContinuousVideoFeed': true,
      'HandleArbitraryRotations': false,
      'DetermineFaceRotationAngle': false,
      'InternalResizeWidth': 256,
      'FaceDetectionThreshold': 5
    });

    // Setting liveness detection parameters
    _tracker.setMultipleParameters({
      'DetectLiveness': true,
      'SmoothAttributeLiveness': true,
      'LivenessFramesCount': 6
    });
  }

  static void _worker(_FacesTrackerIsolateInfo info) async {
    final sendPort = info.port;
    final tracker = Tracker.fromHandle(info.trackerHandle);
    final ids = Int64Buffer.fromInfo(info.idsBufferInfo);

    final receivePort = ReceivePort();
    receivePort.listen((data) {
      var image = Image.fromHandle(data.image);

      final rotation = Platform.isAndroid
          ? data.orientation ~/ 90
          : -(data.orientation ~/ 90) + 1;

      if (rotation != 0) {
        var rotatedImage = image.rotate90(rotation);
        image.free();
        image = rotatedImage;
      }

      if (data.frontFacing && !Platform.isIOS) {
        image.mirror(true);
      }

      ids.length = 0;
      try {
        tracker.feedFrame(0, image, ids: ids);
      } on FaceNotFoundError {
        /*No faces were found*/
      }

      image.free();
      sendPort.send(null);
    });

    sendPort.send(receivePort.sendPort);
  }

  void _initialize() async {
    await _openTracker();

    _receive.listen((msg) async {
      if (msg is SendPort) {
        _send = msg;
        _state = FaceTrackerState.waitingForImage;
        return;
      }

      final imageToSave = _fsdkImage;
      _fsdkImage = null;
      imageToSave?.free();
      _onStateChange(FaceTrackerState.idsReady);
    });

    _isolate = await Isolate.spawn(
      _worker,
      _FacesTrackerIsolateInfo(
        _receive.sendPort,
        _tracker.handle,
        _ids.getInfo(),
      ),
    );
  }

  void process(CameraImage image, int orientation, bool frontFacing) async {
    if (_state == FaceTrackerState.notInitialized) {
      _state = FaceTrackerState.initializing;
      _initialize();
      return;
    }

    if (_state != FaceTrackerState.waitingForImage) {
      return;
    }

    _onStateChange(FaceTrackerState.waitingForIds);
    final frameImage = _converter.convert(image);
    final workerImage = frameImage.copy();
    _fsdkImage?.free();
    _fsdkImage = frameImage;
    _send.send(_WorkerData(workerImage.handle, orientation, frontFacing));
  }

  List<FaceWrapper> faces() {
    if (_state != FaceTrackerState.idsReady) {
      return <FaceWrapper>[];
    }

    return _ids.map((id) => FaceWrapper(id, _tracker)).toList(growable: false);
  }

  void resetTracker() {
    _tracker.clear();
    _setTrackerParameters();
  }

  void setNameForId(int id, String name) {
    _tracker.lockID(id);
    _tracker.setName(id, name);
    _tracker.unlockID(id);
  }

  String getNameForId(int id) {
    _tracker.lockID(id);
    final name = _tracker.getName(id);
    _tracker.unlockID(id);

    return name;
  }

  FaceMatchResult matchFace(Image img) {
    FSDK.FaceTemplate faceTemplate = FSDK.GetFaceTemplate(img);

    var similarityResults = _tracker.matchFaces(
      faceTemplate,
      similarityThreshold,
    );

    if (similarityResults.isNotEmpty) {
      final id = similarityResults[0].id;
      final similarity = similarityResults[0].similarity;
      final FaceWrapper face = FaceWrapper(id, _tracker);
      final double liveness = face.checkLiveness() ? face.liveness : 0.0;

      String name = getNameForId(id);

      return FaceMatchResult(name, id, similarity, liveness, image: img);
    }

    return FaceMatchResult("", -1, 0.0, 0.0);
  }

  void next() {
    if (_state == FaceTrackerState.idsReady) {
      _state = FaceTrackerState.waitingForImage;
    }
  }
}

import 'dart:io';
import 'dart:isolate';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart' hide Image;
import 'package:flutter/services.dart';
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
import 'package:google_mlkit_barcode_scanning/google_mlkit_barcode_scanning.dart';
import 'package:path_provider/path_provider.dart';

class _FacesTrackerIsolateInfo {
  final SendPort port;
  final int trackerHandle;
  final BufferInfo idsBufferInfo;
  final RootIsolateToken rootIsolateToken;

  _FacesTrackerIsolateInfo(
      this.port, this.trackerHandle, this.idsBufferInfo, this.rootIsolateToken);
}

class _WorkerData {
  final int image;
  final int orientation;
  final bool frontFacing;
  final bool enableSaveFrame;

  _WorkerData(
      this.image, this.orientation, this.frontFacing, this.enableSaveFrame);
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
    } catch (_) {
      return false;
    } finally {
      _tracker.unlockID(_id);
    }
    return true;
  }
}

enum FaceTrackerState {
  notInitialized,
  referenceFaceNotDetected,
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
  final String? matchImagePath;

  /// Whether the native tracker found any face at all in the frame this
  /// result was computed from, independent of [similarity]/[liveness] --
  /// those only measure how well a found face matches the reference image,
  /// so they stay 0 both when no face was present and when a face was
  /// present but unmatched. This field disambiguates the two.
  final bool faceDetectedInFrame;

  FaceMatchResult(
    this.name,
    this.id,
    this.similarity,
    this.liveness, {
    this.matchImagePath,
    this.faceDetectedInFrame = false,
  });
}

class FacesTracker extends ChangeNotifier {
  FaceTrackerState _state = FaceTrackerState.notInitialized;
  FaceTrackerState get state => _state;

  bool _enableSaveFrame = true;
  DateTime? _lastProcessAt;

  set enableSaveFrame(bool enable) {
    _enableSaveFrame = enable;
  }

  void _onStateChange(FaceTrackerState newState) {
    _state = newState;
    if (hasListeners) {
      notifyListeners();
    }
  }

  late SendPort _send;
  Isolate? _isolate;
  Image? _fsdkImage;
  int _lastOrientation = 0;
  bool _lastFrontFacing = false;

  late final _tracker = Tracker();
  final _receive = ReceivePort();
  final _converter = ImageConverter();
  final _ids = Int64Buffer.allocate(5);

  int get width => _converter.width;
  int get height => _converter.height;

  double similarityThreshold;

  FacesTracker({
    required this.similarityThreshold,
  });

  @override
  void dispose() {
    _isolate?.kill(priority: Isolate.immediate);
    _converter.free();
    _tracker.free();
    super.dispose();
  }

  Future<void> _initTracker() async {
    // Remove old saved image if exists
    final tempDir = await _getTemporaryFilePath();
    final tempFile = File(tempDir);
    if (await tempFile.exists()) {
      await tempFile.delete();
      debugPrint('Deleted old temporary face image at $tempDir');
    }
    _setTrackerParameters();
  }

  void reset() {
    if (_state == FaceTrackerState.notInitialized) {
      // The worker isolate is only spawned lazily, on the first call to
      // [process] (see below). Forcing _state to waitingForImage here before
      // that happens would skip _initialize() entirely -- the isolate would
      // never be spawned, so no frame is ever processed again for the rest
      // of the test (QR detection and face matching both silently stop
      // working). Let the first [process] call take the normal
      // notInitialized -> initializing -> _initialize() path instead.
      return;
    }
    _tracker.clear();
    _setTrackerParameters();
    _onStateChange(FaceTrackerState.waitingForImage);
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

  void _initialize() async {
    await _initTracker();

    _receive.listen((msg) async {
      if (msg is SendPort) {
        _send = msg;
        _state = FaceTrackerState.waitingForImage;
        return;
      }

      if (!_enableSaveFrame) {
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
        RootIsolateToken.instance!,
      ),
    );
  }

  static void _worker(_FacesTrackerIsolateInfo info) async {
    BackgroundIsolateBinaryMessenger.ensureInitialized(info.rootIsolateToken);

    final sendPort = info.port;
    final tracker = Tracker.fromHandle(info.trackerHandle);
    final ids = Int64Buffer.fromInfo(info.idsBufferInfo);

    final receivePort = ReceivePort();
    receivePort.listen((data) async {
      _WorkerData wData = data as _WorkerData;
      Image image = Image.fromHandle(wData.image);

      image = _normalizeImage(image, wData.orientation, wData.frontFacing);

      ids.length = 0;
      try {
        tracker.feedFrame(0, image, ids: ids);
        if (wData.enableSaveFrame && ids.isNotEmpty) {
          await saveFaceToFile(image);
        }
      } on FaceNotFoundError {
        /*No faces were found*/
      }

      image.free();
      sendPort.send(null);
    });

    sendPort.send(receivePort.sendPort);
  }

  static Image _normalizeImage(Image image, int orientation, bool frontFacing) {
    final rotation = Platform.isAndroid ? orientation ~/ 90 : 0;

    if (rotation != 0) {
      var rotatedImage = image.rotate90(rotation);
      image.free();
      image = rotatedImage;
    }

    if (frontFacing && !Platform.isIOS) {
      image.mirror(true);
    }
    return image;
  }

  /// Saves the frame currently held in memory (the last one handed to
  /// [process]) to file on demand, without requiring [enableSaveFrame] to be
  /// continuously on. Used to fetch a face image once, right when a match is
  /// confirmed, instead of writing to disk on every processed frame (which
  /// causes camera freezes).
  Future<File?> saveCurrentFrame() {
    final image = _fsdkImage;
    if (image == null) {
      return Future.value(null);
    }
    // The frame kept in memory is the raw converted image: unlike the copy
    // sent to the worker isolate, it was never rotated/mirrored for the
    // camera's orientation, so it must go through the same normalization
    // before being saved, otherwise the saved photo comes out rotated.
    final normalized = _normalizeImage(image, _lastOrientation, _lastFrontFacing);
    _fsdkImage = normalized;
    return saveFaceToFile(normalized);
  }

  static Future<File?> saveFaceToFile(Image img) async {
    try {
      final filePath = await _getTemporaryFilePath();
      _normalizeImage(img, 0, false).saveToFile(filePath);
      debugPrint('Face image saved to $filePath');
      return File(filePath);
    } catch (e) {
      debugPrint('Error saving face image: $e');
      return null;
    }
  }

  static Future<String> _getTemporaryFilePath() async {
    final directory = await getApplicationDocumentsDirectory();
    final filePath = '${directory.path}/recognized_face.jpg';
    return filePath;
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

    final now = DateTime.now();
    if (_lastProcessAt != null &&
        now.difference(_lastProcessAt!) < const Duration(milliseconds: 500)) {
      return;
    }
    _lastProcessAt = now;

    _onStateChange(FaceTrackerState.waitingForIds);
    try {
      final frameImage = _converter.convert(image);
      final workerImage = frameImage.copy();
      _fsdkImage?.free();
      _fsdkImage = frameImage;
      _lastOrientation = orientation;
      _lastFrontFacing = frontFacing;
      _send.send(
        _WorkerData(
          workerImage.handle,
          orientation,
          frontFacing,
          _enableSaveFrame,
        ),
      );
    } catch (e) {
      debugPrint('Error processing camera image: $e');
    }
  }

  List<FaceWrapper> faces() {
    if (_state != FaceTrackerState.idsReady) {
      return <FaceWrapper>[];
    }

    return _ids.map((id) => FaceWrapper(id, _tracker)).toList(growable: false);
  }

  /// Whether the last frame fed to [feedFrame] (in the worker isolate)
  /// actually contained a detectable face, regardless of whether it matches
  /// the reference image. [_ids] is populated straight from the native
  /// tracker's per-frame detection, so it stays empty on an empty/no-face
  /// frame even when match reporting is otherwise permissive.
  bool get hasFaceInFrame => _ids.isNotEmpty;

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

  bool findFace(Image img) {
    FSDK.FaceTemplate faceTemplate = FSDK.GetFaceTemplate(img);
    return true;
  }

  Future<FaceMatchResult> matchFace(Image img) async {
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

      return FaceMatchResult(
        name,
        id,
        similarity,
        liveness,
        matchImagePath: await _getTemporaryFilePath(),
        faceDetectedInFrame: true,
      );
    }

    return FaceMatchResult("", -1, 0.0, 0.0, faceDetectedInFrame: hasFaceInFrame);
  }

  void next() {
    if (_state == FaceTrackerState.idsReady) {
      _state = FaceTrackerState.waitingForImage;
    }
  }
}

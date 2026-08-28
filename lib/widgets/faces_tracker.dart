import 'dart:async';
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

/// Bootstrap payload for the worker isolate.
///
/// Deliberately carries no tracker/buffer handle: those only exist once a
/// licence has been activated and a camera session opened, whereas the
/// expensive part of starting the worker (initialising the background binary
/// messenger and loading the native library into the new isolate) does not
/// depend on them at all. Measured at ~7s on device, which is why the worker
/// is started once at app launch and handed its handles later, via
/// [_WorkerSession].
class _FacesTrackerIsolateInfo {
  final SendPort port;
  final RootIsolateToken rootIsolateToken;

  _FacesTrackerIsolateInfo(this.port, this.rootIsolateToken);
}

/// Binds the long-lived worker to one camera session's native objects.
///
/// Sent every time a [FacesTracker] initialises. [sessionId] is echoed back on
/// every result so late frames from a previous session can be discarded
/// instead of being fed to a tracker that has since been freed -- these
/// handles are raw pointers (see `Tracker.fromHandle`), so using a stale one
/// is a segfault, not an exception.
class _WorkerSession {
  final int sessionId;
  final int trackerHandle;
  final BufferInfo idsBufferInfo;

  _WorkerSession(this.sessionId, this.trackerHandle, this.idsBufferInfo);
}

class _WorkerData {
  final int image;
  final int orientation;
  final bool frontFacing;
  final bool enableSaveFrame;
  final int frameId;
  final int sessionId;

  _WorkerData(this.image, this.orientation, this.frontFacing,
      this.enableSaveFrame, this.frameId, this.sessionId);
}

/// Sent back by the worker once [Tracker.feedFrame] has actually run for
/// [frameId], so the main isolate knows precisely which frame the native
/// tracker's state (used by [FacesTracker.matchFace]) now reflects.
class _WorkerResult {
  final int frameId;
  final int sessionId;

  _WorkerResult(this.frameId, this.sessionId);
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

/// TEMPORARY INSTRUMENTATION: prints with a wall-clock timestamp, so tracker
/// state transitions can be lined up against the [FLOW]/[QRPERF] logs when
/// read straight from logcat. Remove with those logs.
void _tsPrint(String message) {
  final DateTime now = DateTime.now();
  String two(int n) => n.toString().padLeft(2, '0');
  String three(int n) => n.toString().padLeft(3, '0');
  final String stamp = '${two(now.hour)}:${two(now.minute)}:'
      '${two(now.second)}.${three(now.millisecond)}';
  // ignore: avoid_print
  print('$stamp $message');
}

/// Owns the single face-tracking worker isolate for the whole app.
///
/// Starting that isolate costs ~7s on device (background binary messenger +
/// native library load), and it used to be paid lazily on the first camera
/// frame of every preview -- so the user waited through it on the QR screen,
/// during onboarding and on the Luxand auth screen alike. It is started once
/// here instead, ideally at app launch (see [FaceWorker.start]), and kept
/// alive until the process dies.
///
/// The isolate holds no per-session state of its own: each [FacesTracker]
/// binds it to its own native tracker via [bindSession], and results are
/// tagged with that session id so anything left in flight from a previous
/// preview is dropped rather than applied to the new one.
class FaceWorker {
  FaceWorker._();

  static final FaceWorker instance = FaceWorker._();

  /// Kept so the isolate is never garbage-collected for the life of the app.
  /// Deliberately never killed: the whole point of this class is that the
  /// worker outlives every individual camera session.
  // ignore: unused_field
  Isolate? _isolate;
  SendPort? _send;
  final ReceivePort _receive = ReceivePort();

  /// Completes once the worker has answered with its [SendPort].
  Completer<void>? _ready;

  int _sessionCounter = 0;
  int _currentSessionId = 0;

  /// Called with (frameId) for results belonging to the current session.
  void Function(int frameId)? _onFrameProcessed;

  /// Whether the worker has been started (it may still be booting).
  bool get isStarted => _ready != null;

  /// Boots the worker. Safe to call more than once: later calls return the
  /// same future. Call this as early as possible -- it needs nothing but a
  /// [RootIsolateToken], so it does not have to wait for a licence, a camera
  /// or a signed-in user.
  Future<void> start() {
    final Completer<void>? existing = _ready;
    if (existing != null) {
      return existing.future;
    }
    final Completer<void> ready = Completer<void>();
    _ready = ready;

    _receive.listen((Object? msg) {
      if (msg is SendPort) {
        _send = msg;
        _tsPrint('[TRACK] worker isolate ready (app-wide)');
        if (!ready.isCompleted) {
          ready.complete();
        }
        return;
      }
      final _WorkerResult result = msg as _WorkerResult;
      // Late frame from a preview that has already gone away: its native
      // tracker may well have been freed, so nothing may act on it.
      if (result.sessionId != _currentSessionId) {
        return;
      }
      _onFrameProcessed?.call(result.frameId);
    });

    _tsPrint('[TRACK] spawning worker isolate (app-wide)...');
    Isolate.spawn(
      _worker,
      _FacesTrackerIsolateInfo(_receive.sendPort, RootIsolateToken.instance!),
    ).then((Isolate isolate) {
      _isolate = isolate;
    }).catchError((Object e) {
      _tsPrint('[TRACK] worker spawn failed: $e');
      if (!ready.isCompleted) {
        ready.completeError(e);
      }
    });

    return ready.future;
  }

  /// Binds the worker to one camera session's native objects and returns that
  /// session's id. Any result still in flight for an earlier session is
  /// discarded from here on.
  Future<int> bindSession({
    required int trackerHandle,
    required BufferInfo idsBufferInfo,
    required void Function(int frameId) onFrameProcessed,
  }) async {
    await start();
    final int sessionId = ++_sessionCounter;
    _currentSessionId = sessionId;
    _onFrameProcessed = onFrameProcessed;
    _send!.send(_WorkerSession(sessionId, trackerHandle, idsBufferInfo));
    _tsPrint('[TRACK] worker bound to session $sessionId');
    return sessionId;
  }

  /// Detaches the current session without stopping the worker, so a preview
  /// being disposed cannot leave a callback pointing at dead state.
  void releaseSession(int sessionId) {
    if (_currentSessionId != sessionId) {
      return;
    }
    _onFrameProcessed = null;
    _currentSessionId = 0;
  }

  /// Sends one frame for processing. Ignored when the worker is not ready or
  /// the session has moved on.
  void feed(_WorkerData data) {
    if (_send == null || data.sessionId != _currentSessionId) {
      return;
    }
    _send!.send(data);
  }

  static void _worker(_FacesTrackerIsolateInfo info) async {
    BackgroundIsolateBinaryMessenger.ensureInitialized(info.rootIsolateToken);

    final SendPort sendPort = info.port;
    final ReceivePort receivePort = ReceivePort();

    // Bound by the first _WorkerSession message and rebound on every later
    // one, so the isolate outlives any individual camera session.
    Tracker? tracker;
    Int64Buffer? ids;
    int boundSession = 0;

    receivePort.listen((Object? data) async {
      if (data is _WorkerSession) {
        tracker = Tracker.fromHandle(data.trackerHandle);
        ids = Int64Buffer.fromInfo(data.idsBufferInfo);
        boundSession = data.sessionId;
        return;
      }

      final _WorkerData wData = data as _WorkerData;
      final Tracker? boundTracker = tracker;
      final Int64Buffer? boundIds = ids;
      // Not bound yet, or a frame from a session this isolate has since been
      // rebound away from. Its handles may already be freed, and these are
      // raw pointers, so touching them would be a segfault rather than an
      // exception: drop the frame and free just the image we were handed.
      if (boundTracker == null ||
          boundIds == null ||
          wData.sessionId != boundSession) {
        Image.fromHandle(wData.image).free();
        return;
      }

      Image image = Image.fromHandle(wData.image);
      image = FacesTracker._normalizeImage(
        image,
        wData.orientation,
        wData.frontFacing,
      );

      boundIds.length = 0;
      try {
        boundTracker.feedFrame(0, image, ids: boundIds);
        if (wData.enableSaveFrame && boundIds.isNotEmpty) {
          await FacesTracker.saveFaceToFile(image);
        }
      } on FaceNotFoundError {
        /*No faces were found*/
      }

      image.free();
      sendPort.send(_WorkerResult(wData.frameId, wData.sessionId));
    });

    sendPort.send(receivePort.sendPort);
  }
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

  /// TEMPORARY INSTRUMENTATION: how many camera frames were dropped because
  /// the tracker was not in `waitingForImage` (remove with the [TRACK] logs).
  int _framesSkipped = 0;

  /// TEMPORARY INSTRUMENTATION: how many frames were skipped while waiting for
  /// the worker to confirm the frame held for capture (remove with the [TRACK]
  /// logs).
  int _framesHeldForCapture = 0;

  /// When the current frame started being held for the worker, so the wait
  /// can be bounded instead of stalling detection forever.
  DateTime? _heldFrameSince;

  set enableSaveFrame(bool enable) {
    _enableSaveFrame = enable;
  }

  void _onStateChange(FaceTrackerState newState) {
    if (newState != _state) {
      _tsPrint('[TRACK] state: ${_state.name} -> ${newState.name}');
    }
    _state = newState;
    if (hasListeners) {
      notifyListeners();
    }
  }

  Image? _fsdkImage;
  int _lastOrientation = 0;
  bool _lastFrontFacing = false;

  /// Identifies which frame is currently held in [_fsdkImage], and which
  /// frame the native tracker's state (as read by [matchFace]) actually
  /// reflects. Incremented on every [process] call, echoed back by the
  /// worker in [_WorkerResult] once [Tracker.feedFrame] has run for it.
  ///
  /// A newer camera frame can arrive and overwrite [_fsdkImage] while a match
  /// triggered by an older frame is still being resolved (matching runs
  /// asynchronously against the reference face, see
  /// [FaceRecognitionPreviewState._onTrackerUpdate]). Comparing these two ids
  /// lets [saveCurrentFrame] detect that drift instead of silently saving a
  /// frame that was never the one matched -- e.g. the user's hand now in
  /// front of their face while moving the device towards the QR reader.
  int _frameId = 0;
  int? _fsdkImageFrameId;
  int? _lastFedFrameId;

  late final _tracker = Tracker();
  final _converter = ImageConverter();
  final _ids = Int64Buffer.allocate(5);

  /// This tracker's slot in the shared [FaceWorker]. Frames and results are
  /// tagged with it, so a frame still in flight when a preview goes away can
  /// never be fed to a native tracker that has since been freed.
  int _sessionId = 0;
  bool _disposed = false;

  /// TEMPORARY INSTRUMENTATION: when each frame was handed to the worker, to
  /// measure the round-trip (remove with the [TRACK] logs).
  final Map<int, DateTime> _frameSentAt = <int, DateTime>{};

  int get width => _converter.width;
  int get height => _converter.height;

  double similarityThreshold;

  FacesTracker({
    required this.similarityThreshold,
  });

  @override
  void dispose() {
    // The worker isolate is shared and app-wide now (see [FaceWorker]), so it
    // deliberately outlives this tracker instead of being killed here. Only
    // this tracker's session is released, which stops any in-flight frame
    // from reaching the native objects freed just below.
    _disposed = true;
    FaceWorker.instance.releaseSession(_sessionId);
    _referenceTemplate?.free();
    _referenceTemplate = null;
    _converter.free();
    _tracker.free();
    super.dispose();
  }

  void reset() {
    if (_state == FaceTrackerState.notInitialized) {
      // This tracker binds to the shared worker lazily, on the first call to
      // [process] (see below). Forcing _state to waitingForImage here before
      // that happens would skip _initialize() entirely -- no session would
      // ever be bound, so no frame is ever processed again for the rest of
      // the test (QR detection and face matching both silently stop working).
      // Let the first [process] call take the normal
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

  void _initialize() async {
    _tsPrint('[TRACK] _initialize() started');
    await _initTracker();
    _tsPrint('[TRACK] _initTracker() done (params set, temp file cleaned)');

    // Binding is near-instant when the worker was started at app launch; it
    // only falls back to paying the ~7s boot cost here if it was not.
    final int sessionId = await FaceWorker.instance.bindSession(
      trackerHandle: _tracker.handle,
      idsBufferInfo: _ids.getInfo(),
      onFrameProcessed: _onFrameProcessed,
    );
    if (_disposed) {
      // Disposed while the worker was still booting: hand the session back so
      // no result is ever routed to this now-dead tracker.
      FaceWorker.instance.releaseSession(sessionId);
      return;
    }
    _sessionId = sessionId;
    _tsPrint('[TRACK] worker bound -> waitingForImage');
    _onStateChange(FaceTrackerState.waitingForImage);
  }

  /// Called by [FaceWorker] once `feedFrame` has actually run for [frameId]:
  /// the native tracker's state (read by [matchFace]) reflects it, so it is
  /// only now safe to notify listeners and let `_onTrackerUpdate` match
  /// against it.
  void _onFrameProcessed(int frameId) {
    if (_disposed) {
      return;
    }
    // TEMPORARY INSTRUMENTATION (remove with the [TRACK] logs): how long the
    // worker took to feed this frame to the native tracker.
    final DateTime? sentAt = _frameSentAt.remove(frameId);
    if (sentAt != null) {
      final int rtt = DateTime.now().difference(sentAt).inMilliseconds;
      if (rtt > 120) {
        _tsPrint('[TRACK] worker round-trip slow: ${rtt}ms (frame $frameId)');
      }
    }
    _lastFedFrameId = frameId;

    if (_enableSaveFrame) {
      final imageToSave = _fsdkImage;
      _fsdkImage = null;
      imageToSave?.free();
    }
    _onStateChange(FaceTrackerState.idsReady);
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
  ///
  /// Returns null if [_fsdkImage] no longer matches the frame the tracker
  /// actually matched (see [_fsdkImageFrameId]/[_lastFedFrameId]) -- saving it
  /// anyway would silently hand back a frame that was never the one checked,
  /// e.g. one grabbed while the user's hand was still moving the device.
  Future<File?> saveCurrentFrame() {
    final image = _fsdkImage;
    if (image == null) {
      return Future.value(null);
    }
    if (_fsdkImageFrameId != _lastFedFrameId) {
      print(
        'Discarding stale frame for capture: held frame $_fsdkImageFrameId, '
        'tracker matched against frame $_lastFedFrameId',
      );
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
      _framesSkipped++;
      if (_framesSkipped % 20 == 1) {
        _tsPrint(
          '[TRACK] frame skipped (state=${_state.name}, '
          'skipped=$_framesSkipped)',
        );
      }
      return;
    }

    final now = DateTime.now();
    if (_lastProcessAt != null &&
        now.difference(_lastProcessAt!) < const Duration(milliseconds: 500)) {
      return;
    }

    // Hold the frame currently kept for capture until the worker has actually
    // fed it to the native tracker.
    //
    // [saveCurrentFrame] refuses to save a frame the tracker never matched
    // against (it compares these two ids), and the worker round-trip was
    // measured at 150-660ms while frames are accepted every 500ms. So the
    // held frame was permanently one ahead of the matched one, every capture
    // failed its freshness check, and every face match was then dropped by
    // the caller -- the test ended with "your face was not detected" despite
    // the face being recognised on five consecutive cycles.
    //
    // Skipping here costs nothing: those frames were being captured and
    // thrown away anyway, and it paces intake to what the worker can absorb.
    if (_fsdkImageFrameId != null && _fsdkImageFrameId != _lastFedFrameId) {
      // Bounded wait: a result can legitimately never arrive (the session was
      // rebound, the worker dropped the frame), and blocking on it forever
      // would freeze detection for the rest of the test. Past this delay the
      // frame is replaced as before -- back to the old behaviour rather than
      // a stall.
      final DateTime? heldSince = _heldFrameSince;
      if (heldSince != null &&
          now.difference(heldSince) > const Duration(seconds: 2)) {
        _tsPrint(
          '[TRACK] frame hold timed out (frame $_fsdkImageFrameId never '
          'confirmed, last fed $_lastFedFrameId) -- releasing',
        );
        _heldFrameSince = null;
      } else {
        _heldFrameSince ??= now;
        _framesHeldForCapture++;
        if (_framesHeldForCapture % 20 == 1) {
          _tsPrint(
            '[TRACK] frame held (awaiting worker for frame $_fsdkImageFrameId, '
            'last fed $_lastFedFrameId, held=$_framesHeldForCapture)',
          );
        }
        return;
      }
    }
    _heldFrameSince = null;
    _lastProcessAt = now;

    // Note: [_state] is only flipped to [waitingForIds] here, without a call
    // to [_onStateChange] (so without notifying listeners yet). Notifying
    // now would let [_onTrackerUpdate] call [matchFace] before [feedFrame]
    // has run for this frame in the worker isolate, matching against the
    // native tracker's *previous* state instead. Listeners are notified once
    // the worker echoes back that this frame was actually fed (see
    // [_initialize]'s receive-port handler).
    _state = FaceTrackerState.waitingForIds;
    try {
      // TEMPORARY INSTRUMENTATION (remove with the [TRACK] logs): convert()
      // and copy() both run on the UI thread for every frame fed to the
      // worker.
      final Stopwatch swConvert = Stopwatch()..start();
      final frameImage = _converter.convert(image);
      final workerImage = frameImage.copy();
      swConvert.stop();
      if (swConvert.elapsedMilliseconds > 30) {
        _tsPrint(
          '[TRACK] frame convert+copy slow: ${swConvert.elapsedMilliseconds}ms',
        );
      }
      final frameId = ++_frameId;
      _fsdkImage?.free();
      _fsdkImage = frameImage;
      _fsdkImageFrameId = frameId;
      _lastOrientation = orientation;
      _lastFrontFacing = frontFacing;
      _frameSentAt[frameId] = DateTime.now();
      FaceWorker.instance.feed(
        _WorkerData(
          workerImage.handle,
          orientation,
          frontFacing,
          _enableSaveFrame,
          frameId,
          _sessionId,
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

  /// Cached template of the reference image passed to [matchFace].
  ///
  /// [img] is the *reference* photo (the registered employee's picture, see
  /// `FaceRecognitionPreviewState._userFace`), not the camera frame -- the
  /// camera frames go to the native tracker through [process]/`feedFrame`,
  /// and `matchFaces` compares this template against what the tracker has
  /// seen. That reference never changes during a test, yet `GetFaceTemplate`
  /// was being run on it for every single frame: measured at ~300ms each
  /// time, on the UI thread, which starved the tracker of frames (match
  /// cycles were ~1s apart, so it kept losing tracking and re-detecting).
  /// Computed once here and reused; [clearReferenceTemplate] drops it when
  /// the reference image itself changes.
  FSDK.FaceTemplate? _referenceTemplate;
  int? _referenceTemplateFor;

  /// Drops the cached reference template, so the next [matchFace] recomputes
  /// it. Call when the reference image is reloaded.
  void clearReferenceTemplate() {
    _referenceTemplate?.free();
    _referenceTemplate = null;
    _referenceTemplateFor = null;
  }

  Future<FaceMatchResult> matchFace(Image img) async {
    // Recompute only when the reference image actually changed (its native
    // handle identifies it): otherwise reuse the cached template.
    if (_referenceTemplate == null || _referenceTemplateFor != img.handle) {
      final Stopwatch swTemplate = Stopwatch()..start();
      _referenceTemplate?.free();
      _referenceTemplate = FSDK.GetFaceTemplate(img);
      _referenceTemplateFor = img.handle;
      swTemplate.stop();
      _tsPrint(
        '[TRACK] reference template computed in '
        '${swTemplate.elapsedMilliseconds}ms (cached from now on)',
      );
    }
    final FSDK.FaceTemplate faceTemplate = _referenceTemplate!;

    final Stopwatch swMatch = Stopwatch()..start();
    var similarityResults = _tracker.matchFaces(
      faceTemplate,
      similarityThreshold,
    );
    swMatch.stop();
    if (swMatch.elapsedMilliseconds > 40) {
      _tsPrint(
        '[TRACK] matchFace slow: matchFaces=${swMatch.elapsedMilliseconds}ms '
        'results=${similarityResults.length}',
      );
    }

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

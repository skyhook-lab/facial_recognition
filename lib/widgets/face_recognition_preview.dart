import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter_face_sdk/flutter_face_sdk.dart' as FSDK;
import 'package:flutter_face_sdk/widgets/faces_painter.dart';
import 'package:flutter_face_sdk/widgets/faces_tracker.dart';
import 'package:google_mlkit_barcode_scanning/google_mlkit_barcode_scanning.dart';

/// TEMPORARY INSTRUMENTATION: prints with a wall-clock timestamp.
///
/// The in-app log export (`AppLogEntry.format`) already stamps every line, but
/// these are read straight from logcat during perf work, where Flutter's own
/// `I/flutter` lines carry no time of their own.
void _tsPrint(String message) {
  final DateTime now = DateTime.now();
  String two(int n) => n.toString().padLeft(2, '0');
  String three(int n) => n.toString().padLeft(3, '0');
  final String stamp = '${two(now.hour)}:${two(now.minute)}:'
      '${two(now.second)}.${three(now.millisecond)}';
  print('$stamp $message');
}

const luxandURL = 'https://www.luxand.com/facesdk';
const luxandSwatch = MaterialColor(0xff5a95dd, <int, Color>{
  50: Color(0xffa4c4ec),
  100: Color(0xff9cbfea),
  200: Color(0xff8bb4e7),
  300: Color(0xff7baae3),
  400: Color(0xff6a9fe0),
  500: Color(0xff5a95dd),
  600: Color(0xff5187c7),
  700: Color(0xff4878b1),
  800: Color(0xff3f699b),
  900: Color(0xff365a85)
});

bool _initialized = false;
late List<CameraDescription> _cameras;

Future<void> initializeLuxand(String licenseKey) async {
  if (!_initialized) {
    _cameras = await availableCameras();
    FSDK.ActivateLibrary(licenseKey);
    FSDK.InitializeLibrary();
    _initialized = true;
  }
}

typedef OnInitializationError = void Function(Object error);
typedef OnInitialized = void Function();

class FaceRecognitionPreview extends StatefulWidget {
  final OnInitializationError onInitializationError;
  final ValueChanged<FaceMatchResult> onMatchResult;
  final ValueChanged<FaceTrackerState> onTrackerStateChanged;
  final OnInitialized? onInitialized;
  final String userFacePath;
  final String licenseKey;
  final bool enabled;
  final bool enableSaveFrame;
  final bool displayDebugInfo;
  final double similarityThreshold;

  /// Whether to scan camera frames for a QR code. Independent of the face
  /// checks: scanning starts with the first camera frame, without waiting for
  /// the tracker to come up.
  final bool scanQrCode;

  /// Whether a match with similarity 0 (no reference match found by the
  /// native tracker) should still be reported to [onMatchResult] instead of
  /// being silently dropped. Set this when the caller does not require the
  /// detected face to match the reference image (e.g. a guest test).
  final bool reportUnmatchedFaces;

  const FaceRecognitionPreview({
    required this.licenseKey,
    required this.userFacePath,
    required this.onInitializationError,
    required this.onTrackerStateChanged,
    required this.onMatchResult,
    required this.similarityThreshold,
    required this.enableSaveFrame,
    this.enabled = true,
    this.displayDebugInfo = false,
    this.reportUnmatchedFaces = false,
    this.scanQrCode = false,
    this.onInitialized,
    super.key,
  });

  @override
  FaceRecognitionPreviewState createState() => FaceRecognitionPreviewState();
}

class FaceRecognitionPreviewState extends State<FaceRecognitionPreview>
    with WidgetsBindingObserver {
  /// Number of preview States ever created in this process.
  ///
  /// Static on purpose: the camera failures we are chasing ("No supported
  /// surface combination", "Unable to open camera") are suspected to come from
  /// two *different* State instances overlapping -- one still tearing its
  /// camera session down while the next one binds a new set of use cases. A
  /// per-instance flag cannot see that, so every lifecycle log below carries
  /// [_instanceId] to make the interleaving visible in the exported logs.
  static int _instanceCount = 0;
  final int _instanceId = ++_instanceCount;

  late final FacesTracker _tracker;
  late FacesPainter _painter;
  late CameraController _controller;
  late FSDK.Image _userFace;
  FaceTrackerState _trackerState = FaceTrackerState.notInitialized;
  bool _isInitializing = false;
  Object? _initError;
  bool _trackerInitialized = false;
  CameraImage? _lastCameraImage;
  BarcodeScanner? _barcodeScanner;

  /// Minimum delay between two QR scans. ML Kit inference costs 80-900ms on
  /// device, so scanning every frame would queue work faster than it drains.
  static const Duration _qrScanInterval = Duration(milliseconds: 400);

  /// Whether a self-driven QR scan is currently running.
  bool _qrScanInFlight = false;
  DateTime? _lastQrScanStartedAt;

  /// Code read by the most recent completed scan, or null when that scan
  /// found none. Read by the host widget on its own cycle.
  String? lastQrCode;

  /// TEMPORARY INSTRUMENTATION: how many tracker notifications were dropped
  /// before the tracker was ready, and whether the first real match cycle has
  /// been logged (remove with the [TRACK] logs).
  int _trackerUpdatesBlocked = 0;
  bool _firstMatchCycleLogged = false;
  DateTime? _lastMatchCycleAt;

  /// TEMPORARY INSTRUMENTATION: timestamps used by the [QRPERF] logs to
  /// report the real interval between QR scans and between camera frames
  /// (remove together with those logs).
  DateTime? _lastQrScanAt;
  DateTime? _lastFrameAt;

  /// Whether [_controller] currently points at a live, initialized camera
  /// session that is safe to build a [CameraPreview] from.
  ///
  /// Tracked separately from `_controller.value.isInitialized`: that value
  /// lags behind the actual dispose (see [_closeCamera]), so a rebuild
  /// landing in that gap -- e.g. triggered by a `MediaQuery` change when the
  /// Android notification shade is dragged down and released quickly --
  /// could still read `isInitialized == true` and build a `CameraPreview` on
  /// a controller that is already disposed, throwing
  /// "buildPreview() was called on a disposed CameraController". This flag
  /// flips false synchronously the instant a close starts.
  bool _controllerReady = false;

  /// The in-flight close operation, if any, so overlapping calls (e.g. the
  /// app going `inactive` while `initialize()` is already mid-flight) await
  /// the same dispose instead of racing to close/dispose [_controller] twice.
  Future<void>? _closingFuture;

  @override
  void initState() {
    super.initState();

    debugPrint('[CAM][#$_instanceId] initState');
    WidgetsBinding.instance.addObserver(this);

    initialize();
  }

  @override
  void didUpdateWidget(covariant FaceRecognitionPreview oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (_trackerInitialized) {
      if (oldWidget.similarityThreshold != widget.similarityThreshold) {
        _tracker.similarityThreshold = widget.similarityThreshold;
      }
      _tracker.enableSaveFrame = widget.enableSaveFrame;
    }
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (!_controller.value.isInitialized) {
      return;
    }

    if (state == AppLifecycleState.inactive) {
      _closeCamera();
      return;
    }

    if (state == AppLifecycleState.resumed) {
      initialize();
      return;
    }
  }

  Future<void> initialize() async {
    debugPrint('[CAM][#$_instanceId] initialize() called');
    if (_isInitializing) {
      debugPrint(
        '[CAM][#$_instanceId] Camera is already opening, skipping '
        'initialization.',
      );
      return;
    }
    if (mounted) {
      setState(() {
        _isInitializing = true;
        _initError = null;
      });
    }

    try {
      await initializeLuxand(widget.licenseKey);
      debugPrint('[Face] <Init> Luxand library initialized.');
      if (!_trackerInitialized) {
        _tracker = FacesTracker(similarityThreshold: 0.01);
        _tracker.enableSaveFrame = widget.enableSaveFrame;
        _tracker.addListener(_onTrackerUpdate);
        _controller = _getCameraController();
        _painter = FacesPainter(
          _tracker,
          _controller,
          drawFPS: widget.displayDebugInfo,
          drawFeatures: widget.displayDebugInfo,
          drawFacesRects: widget.displayDebugInfo,
        );
        _trackerInitialized = true;
      }
      debugPrint('[Face] <Init> Tracker initialized.');

      await _initUserFace();
      debugPrint('[Face] <Init> User face loaded.');

      await _closeCamera();
      _controller = _getCameraController();
      debugPrint('[CAM][#$_instanceId] binding camera use cases...');
      await _controller.initialize();
      debugPrint('[CAM][#$_instanceId] camera bound, starting image stream');
      await _controller.startImageStream(_process);
      debugPrint(
        '[CAM][#$_instanceId] [Face] <Init> Camera initialized and image '
        'stream started.',
      );
      if (mounted) {
        setState(() {
          _controllerReady = true;
        });
      } else {
        _controllerReady = true;
      }

      widget.onInitialized?.call();
    } catch (e, s) {
      if (e is FSDK.FaceNotFoundError) {
        WidgetsBinding.instance.addPostFrameCallback((_) {
          if (mounted) {
            widget.onTrackerStateChanged(
              FaceTrackerState.referenceFaceNotDetected,
            );
          }
        });
        return;
      }

      debugPrint('[CAM][#$_instanceId] BIND FAILED: $e\n$s');
      _initError = e;
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (mounted) {
          widget.onInitializationError(e);
        }
      });
    } finally {
      debugPrint('[Face] <Init> Initialization process completed.');
      if (mounted) {
        setState(() {
          _isInitializing = false;
        });
      }
    }
  }

  Future<void> _initUserFace() async {
    final XFile image = XFile(widget.userFacePath);
    _userFace = FSDK.Image.fromFile(image.path);
    // A fresh reference image invalidates the template cached from the
    // previous one (this runs again on every initialize(), e.g. when the app
    // comes back to the foreground).
    _tracker.clearReferenceTemplate();
    await _tracker.findFace(_userFace);
  }

  String _resolveCameraErrorMessage(Object error) {
    if (error is FSDK.FaceNotFoundError) {
      return 'No face was found in the reference image.';
    }

    return error.toString();
  }

  Future<void> _onTrackerUpdate() async {
    if (_trackerState != _tracker.state) {
      _trackerState = _tracker.state;
      widget.onTrackerStateChanged(_trackerState);
    }
    if (!widget.enabled) {
      return;
    }
    if (![FaceTrackerState.idsReady, FaceTrackerState.waitingForIds]
        .contains(_trackerState)) {
      // TEMPORARY INSTRUMENTATION (remove with the [TRACK] logs): this is the
      // gate that kept the QR result from being consumed for ~8s.
      _trackerUpdatesBlocked++;
      if (_trackerUpdatesBlocked % 10 == 1) {
        _tsPrint(
          '[TRACK] onTrackerUpdate blocked (state=${_trackerState.name}, '
          'count=$_trackerUpdatesBlocked)',
        );
      }
      return;
    }
    if (!_firstMatchCycleLogged) {
      _firstMatchCycleLogged = true;
      _tsPrint('[TRACK] FIRST match cycle running (state=${_trackerState.name})');
    }
    // TEMPORARY INSTRUMENTATION (remove with the [TRACK] logs): the interval
    // between match cycles is what the native tracker actually gets to work
    // with -- ContinuousVideoFeed needs a steady stream, and a choppy one
    // makes it lose tracking and re-detect the face from scratch.
    final DateTime cycleNow = DateTime.now();
    final int sinceLastCycle = _lastMatchCycleAt == null
        ? 0
        : cycleNow.difference(_lastMatchCycleAt!).inMilliseconds;
    _lastMatchCycleAt = cycleNow;

    final Stopwatch swFind = Stopwatch()..start();
    final FaceMatchResult match = await _findMatch(_userFace);
    swFind.stop();

    // `scanQrCode` is only set during the QR phase, so its absence marks the
    // facial phase from inside the SDK, which does not know about the app's
    // EnabledFeatures enum.
    if (!widget.scanQrCode) {
      _tsPrint(
        '[TRACK] face cycle: gap=${sinceLastCycle}ms '
        'findMatch=${swFind.elapsedMilliseconds}ms '
        'similarity=${match.similarity.toStringAsFixed(3)} '
        'faceInFrame=${match.faceDetectedInFrame} '
        'liveness=${match.liveness.toStringAsFixed(3)}',
      );
    }

    if (match.similarity <= 0 && !widget.reportUnmatchedFaces) {
      debugPrint('No match found for the reference face.');
      return;
    }
    widget.onMatchResult(match);
  }

  Future<FaceMatchResult> _findMatch(FSDK.Image img) async {
    try {
      return await _tracker.matchFace(img);
    } on FSDK.FaceNotFoundError {
      return FaceMatchResult("", -1, 0, 0, faceDetectedInFrame: false);
    }
  }

  void reset() {
    if (!_trackerInitialized) {
      return;
    }
    _tracker.reset();
  }

  /// Saves the frame currently held by the tracker to file on demand. Use
  /// this instead of `enableSaveFrame: true` when the file is only needed
  /// once (e.g. right when a match is confirmed), to avoid the camera
  /// freezes caused by saving on every processed frame.
  Future<File?> captureCurrentFrame() => _tracker.saveCurrentFrame();

  /// Scans the latest raw camera frame for a QR code, without going through
  /// the face SDK's save-to-disk pipeline (JPEG encode + file write/read),
  /// which is too expensive to run on every processed frame. This lets QR
  /// detection run continuously (needed since a code may appear on any
  /// frame) without the camera freezes that continuous `enableSaveFrame`
  /// caused.
  Future<String?> detectQrCode() async {
    // TEMPORARY INSTRUMENTATION (remove once the QR-phase stutter is
    // diagnosed): times the two halves of a scan separately, so we can tell
    // whether the cost is the Dart-side frame conversion (runs on the UI
    // thread) or the ML Kit inference behind the platform channel.
    final Stopwatch swTotal = Stopwatch()..start();

    final Stopwatch swConvert = Stopwatch()..start();
    final InputImage? inputImage = _buildBarcodeInputImage();
    swConvert.stop();

    if (inputImage == null) {
      _tsPrint('[QRPERF] no frame available (convert took ${swConvert.elapsedMicroseconds}us)');
      return null;
    }

    // Reused across calls: creating a new BarcodeScanner per frame forces
    // ML Kit to reload its dynamite module and rebuild the TFLite XNNPack
    // delegate every time (visible in logcat as repeated "Replacing N out of
    // N node(s) with delegate" lines), which was the actual source of the
    // per-frame lag causing liveness detection to time out.
    final scanner = _barcodeScanner ??= BarcodeScanner(
      formats: [BarcodeFormat.qrCode],
    );

    final Stopwatch swScan = Stopwatch()..start();
    final barcodes = await scanner.processImage(inputImage);
    swScan.stop();
    swTotal.stop();

    final CameraImage? frame = _lastCameraImage;
    final Duration sinceLast = _lastQrScanAt == null
        ? Duration.zero
        : DateTime.now().difference(_lastQrScanAt!);
    _lastQrScanAt = DateTime.now();

    _tsPrint(
      '[QRPERF] convert=${swConvert.elapsedMilliseconds}ms '
      'mlkit=${swScan.elapsedMilliseconds}ms '
      'total=${swTotal.elapsedMilliseconds}ms '
      'gapSinceLastScan=${sinceLast.inMilliseconds}ms '
      'size=${frame?.width}x${frame?.height} '
      'format=${frame?.format.group} '
      'found=${barcodes.isNotEmpty}',
    );

    return barcodes.isEmpty ? null : barcodes.first.rawValue;
  }

  InputImage? _buildBarcodeInputImage() {
    final CameraImage? image = _lastCameraImage;
    if (image == null) {
      return null;
    }

    final InputImageRotation rotation = InputImageRotationValue.fromRawValue(
          _controller.description.sensorOrientation,
        ) ??
        InputImageRotation.rotation0deg;

    Uint8List bytes;
    InputImageFormat format;
    int bytesPerRow;
    switch (image.format.group) {
      case ImageFormatGroup.yuv420:
        if (Platform.isIOS) {
          format = InputImageFormat.yuv420;
          bytes = image.planes.length == 1
              ? image.planes.first.bytes
              : Uint8List.fromList(
                  image.planes.expand((plane) => plane.bytes).toList(),
                );
          bytesPerRow = image.planes.first.bytesPerRow;
        } else {
          // The Android ML Kit plugin only accepts NV21 or YV12 for raw
          // bytes (it rejects YUV_420_888 outright), so the 3-plane camera
          // buffer must be converted to NV21 first.
          format = InputImageFormat.nv21;
          bytes = _yuv420ToNv21(image);
          bytesPerRow = image.width;
        }
        break;
      case ImageFormatGroup.nv21:
        format = InputImageFormat.nv21;
        bytes = image.planes.first.bytes;
        bytesPerRow = image.planes.first.bytesPerRow;
        break;
      case ImageFormatGroup.bgra8888:
        format = InputImageFormat.bgra8888;
        bytes = image.planes.first.bytes;
        bytesPerRow = image.planes.first.bytesPerRow;
        break;
      case ImageFormatGroup.jpeg:
      case ImageFormatGroup.unknown:
        return null;
    }

    return InputImage.fromBytes(
      bytes: bytes,
      metadata: InputImageMetadata(
        size: Size(image.width.toDouble(), image.height.toDouble()),
        rotation: rotation,
        format: format,
        bytesPerRow: bytesPerRow,
      ),
    );
  }

  /// Converts a 3-plane YUV_420_888 [CameraImage] (Android's default camera
  /// stream format) into a single NV21 buffer, respecting each plane's row
  /// stride and pixel stride since Android devices commonly pad rows or
  /// interleave the U/V planes differently.
  static Uint8List _yuv420ToNv21(CameraImage image) {
    final int width = image.width;
    final int height = image.height;
    final int ySize = width * height;
    final Uint8List nv21 = Uint8List(ySize + (width * height ~/ 2));

    final Plane yPlane = image.planes[0];
    final int yRowStride = yPlane.bytesPerRow;
    if (yRowStride == width) {
      nv21.setRange(0, ySize, yPlane.bytes);
    } else {
      int destOffset = 0;
      for (int row = 0; row < height; row++) {
        final int srcOffset = row * yRowStride;
        nv21.setRange(destOffset, destOffset + width, yPlane.bytes, srcOffset);
        destOffset += width;
      }
    }

    final Plane uPlane = image.planes[1];
    final Plane vPlane = image.planes[2];
    final int uvRowStride = vPlane.bytesPerRow;
    final int uvPixelStride = vPlane.bytesPerPixel ?? 1;
    final int chromaWidth = width ~/ 2;
    final int chromaHeight = height ~/ 2;

    int destOffset = ySize;
    for (int row = 0; row < chromaHeight; row++) {
      final int srcRowOffset = row * uvRowStride;
      for (int col = 0; col < chromaWidth; col++) {
        final int srcOffset = srcRowOffset + col * uvPixelStride;
        // NV21 interleaves chroma as V,U (as opposed to NV12's U,V).
        nv21[destOffset++] = vPlane.bytes[srcOffset];
        nv21[destOffset++] = uPlane.bytes[srcOffset];
      }
    }

    return nv21;
  }

  /// Runs a QR scan straight off the camera stream, independently of the face
  /// tracker's own cycle.
  ///
  /// The scan used to be driven from [_onTrackerUpdate], which only starts
  /// emitting once the native tracker reaches `idsReady`/`waitingForIds`.
  /// That took ~9s from camera start in production logs, and during all that
  /// time no QR scan was attempted at all -- the code was on screen, ML Kit
  /// was warm, and nothing was looking at it. Nothing about reading a QR code
  /// needs a face, so it is driven from the raw frame callback instead, which
  /// runs from the very first frame.
  ///
  /// Results land in [lastQrCode] for the host widget to consume on its next
  /// match cycle, exactly as before -- this changes when scanning starts, not
  /// how a result is reported.
  void _maybeScanQrCode() {
    if (!widget.enabled || !widget.scanQrCode || _qrScanInFlight) {
      return;
    }
    final DateTime now = DateTime.now();
    if (_lastQrScanStartedAt != null &&
        now.difference(_lastQrScanStartedAt!) < _qrScanInterval) {
      return;
    }
    _lastQrScanStartedAt = now;
    _qrScanInFlight = true;
    unawaited(
      detectQrCode()
          .then<void>((String? code) {
            if (mounted) {
              if (code != null && lastQrCode == null) {
                _tsPrint('[TRACK] QR code READ by preview: $code');
              }
              lastQrCode = code;
            }
          })
          .catchError((Object e) {
            debugPrint('[CAM][#$_instanceId] QR scan failed: $e');
          })
          .whenComplete(() => _qrScanInFlight = false),
    );
  }

  void _process(CameraImage image) {
    _lastCameraImage = image;
    _maybeScanQrCode();

    // TEMPORARY INSTRUMENTATION (remove with the [QRPERF] logs): the camera
    // delivers frames at a steady rate, so the interval between calls here is
    // a direct measure of the stutter -- a spike means the UI thread was
    // blocked and frames were dropped. Only slow frames are logged, to avoid
    // the logging itself becoming the bottleneck.
    final DateTime now = DateTime.now();
    if (_lastFrameAt != null) {
      final int gap = now.difference(_lastFrameAt!).inMilliseconds;
      if (gap > 120) {
        _tsPrint('[QRPERF] slow frame: ${gap}ms since previous camera frame');
      }
    }
    _lastFrameAt = now;

    _tracker.process(
      image,
      _controller.description.sensorOrientation,
      _controller.description.lensDirection == CameraLensDirection.front,
    );
  }

  @override
  Widget build(BuildContext context) {
    if (_initError != null) {
      return _buildInitError();
    }

    if (_isInitializing || !_controllerReady) {
      return _buildProgress();
    }
    return Stack(
      fit: StackFit.expand,
      children: <Widget>[
        _buildCameraPreview(),
        Positioned.fill(child: CustomPaint(painter: _painter)),
      ],
    );
  }

  Widget _buildCameraPreview() {
    final Size? previewSize = _controller.value.previewSize;

    if (previewSize == null) {
      return CameraPreview(_controller);
    }

    return ClipRect(
      child: FittedBox(
        fit: BoxFit.cover,
        child: SizedBox(
          width: previewSize.height,
          height: previewSize.width,
          child: CameraPreview(_controller),
        ),
      ),
    );
  }

  Widget _buildProgress() => const Center(child: CircularProgressIndicator());

  Widget _buildInitError() {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Text('Unable to open camera'),
          const SizedBox(height: 8),
          Text(
            _resolveCameraErrorMessage(_initError!),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 12),
          ElevatedButton(
            onPressed: _isInitializing ? null : initialize,
            child: const Text('Retry'),
          ),
        ],
      ),
    );
  }

  CameraController _getCameraController() {
    return CameraController(
      _getCamera(),
      ResolutionPreset.high,
      enableAudio: false,
    );
  }

  CameraDescription _getCamera() {
    return _cameras.firstWhere(
      (element) => element.lensDirection == CameraLensDirection.front,
      orElse: () => _cameras[0],
    );
  }

  Future<void> _closeCamera() {
    return _closingFuture ??= _doCloseCamera().whenComplete(() {
      _closingFuture = null;
    });
  }

  Future<void> _doCloseCamera() async {
    debugPrint('[CAM][#$_instanceId] closing camera (start)');
    // Flip this before the first `await` below so a rebuild racing this close
    // (e.g. from a lifecycle-triggered `MediaQuery` change) never sees a
    // stale `isInitialized == true` and tries to build a preview from the
    // controller we are about to dispose.
    if (mounted) {
      setState(() {
        _controllerReady = false;
      });
    } else {
      _controllerReady = false;
    }

    if (!_controller.value.isInitialized) {
      return;
    }

    if (_controller.value.isStreamingImages) {
      await _controller.stopImageStream();
    }

    await _controller.dispose();
    debugPrint('[CAM][#$_instanceId] closing camera (done)');
  }

  @override
  void dispose() {
    debugPrint('[CAM][#$_instanceId] dispose');
    WidgetsBinding.instance.removeObserver(this);

    _tracker.removeListener(_onTrackerUpdate);
    _tracker.dispose();
    _controller.dispose();
    _barcodeScanner?.close();

    super.dispose();
  }
}

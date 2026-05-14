import 'dart:async';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter_face_sdk/flutter_face_sdk.dart' as FSDK;
import 'package:flutter_face_sdk/widgets/faces_painter.dart';
import 'package:flutter_face_sdk/widgets/faces_tracker.dart';
import 'package:permission_handler/permission_handler.dart';

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
  final bool displayDebugInfo;
  final double similarityThreshold;

  const FaceRecognitionPreview({
    required this.licenseKey,
    required this.userFacePath,
    required this.onInitializationError,
    required this.onTrackerStateChanged,
    required this.onMatchResult,
    required this.similarityThreshold,
    this.enabled = true,
    this.displayDebugInfo = false,
    this.onInitialized,
    super.key,
  });

  @override
  State<StatefulWidget> createState() => _FaceRecognitionPreviewState();
}

class _FaceRecognitionPreviewState extends State<FaceRecognitionPreview>
    with WidgetsBindingObserver {
  late final FacesTracker _tracker;
  late FacesPainter _painter;
  late CameraController _controller;
  late FSDK.Image _userFace;
  FaceTrackerState _trackerState = FaceTrackerState.notInitialized;
  bool _isInitializing = false;
  Object? _initError;
  bool _trackerInitialized = false;

  @override
  void initState() {
    super.initState();

    WidgetsBinding.instance.addObserver(this);

    initialize();
  }

  @override
  void didUpdateWidget(covariant FaceRecognitionPreview oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.similarityThreshold != widget.similarityThreshold) {
      _tracker.similarityThreshold = widget.similarityThreshold;
    }
    _tracker.enableSaveFrame = widget.enabled;
  }

  Future<void> _requestStoragePermission() async {
    final status = await Permission.storage.status;
    if (status.isGranted) {
      return;
    }

    final result = await Permission.storage.request();
    if (result.isGranted != true) {
      throw FSDK.PluginNoPermissionError(
        'Storage permission is required to load the reference face image.',
      );
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
    if (_isInitializing) {
      debugPrint('Camera is already opening, skipping initialization.');
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

      await _requestStoragePermission();
      debugPrint('[Face] <Init> Storage permission granted.');

      await _initUserFace();
      debugPrint('[Face] <Init> User face loaded.');

      await _closeCamera();
      _controller = _getCameraController();
      await _controller.initialize();
      await _controller.startImageStream(_process);
      debugPrint('[Face] <Init> Camera initialized and image stream started.');

      widget.onInitialized?.call();
    } catch (e, s) {
      debugPrint('[Face] <Init> Error during initialization: $e\n$s');
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
    if (_trackerState != FaceTrackerState.idsReady) {
      return;
    }
    final FaceMatchResult match = await _findMatch(_userFace);
    if (match.similarity <= 0) {
      debugPrint('No match found for the reference face.');
      return;
    }
    widget.onMatchResult(match);
  }

  Future<FaceMatchResult> _findMatch(FSDK.Image img) async {
    try {
      return await _tracker.matchFace(img);
    } on FSDK.FaceNotFoundError {
      return FaceMatchResult("", -1, 0, 0);
    }
  }

  void _process(CameraImage image) {
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

    if (_isInitializing || !_controller.value.isInitialized) {
      return _buildProgress();
    }
    return Stack(
      alignment: Alignment.center,
      children: <Widget>[
        _buildCameraPreview(),
        CustomPaint(painter: _painter, child: Container())
      ],
    );
  }

  Widget _buildCameraPreview() {
    return LayoutBuilder(builder: (context, constraints) {
      Widget child = AspectRatio(
        aspectRatio: 1,
        child: FittedBox(
          fit: BoxFit.cover,
          clipBehavior: Clip.hardEdge,
          child: SizedBox(
            height: constraints.maxHeight * _controller.value.aspectRatio,
            width: constraints.maxWidth,
            child: CameraPreview(_controller),
          ),
        ),
      );
      return child;
    });
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

  Future<void> _closeCamera() async {
    if (!_controller.value.isInitialized) {
      return;
    }

    if (_controller.value.isStreamingImages) {
      await _controller.stopImageStream();
    }

    await _controller.dispose();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);

    _tracker.removeListener(_onTrackerUpdate);
    _tracker.dispose();
    _controller.dispose();

    super.dispose();
  }
}

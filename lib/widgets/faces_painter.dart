import 'dart:io';
import 'dart:math';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter_face_sdk/flutter_face_sdk.dart' as FSDK;

import 'faces_tracker.dart' show FacesTracker;

class FaceRect extends Rect {
  final int _id;
  int get id => _id;

  FaceRect(int id, super.a, super.b)
      : _id = id,
        super.fromPoints();
}

class FacesPainter extends CustomPainter {
  static final _paintGreen = Paint()..color = Colors.green;
  static final _paintBlue = Paint()
    ..color = Colors.blue
    ..style = PaintingStyle.stroke
    ..strokeWidth = 3;

  final _faces = <FaceRect>[];
  final _stopwatch = Stopwatch()..start();
  final _textPainter = TextPainter(textDirection: TextDirection.ltr);

  final FacesTracker _tracker;
  final CameraController _controller;
  final bool _drawFPS;
  final bool _drawFeatures;
  final bool _drawFacesRects;
  final bool _drawLiveness;

  FacesPainter(
    FacesTracker tracker,
    CameraController controller, {
    bool drawFPS = false,
    bool drawFeatures = false,
    bool drawFacesRects = false,
    bool drawLiveness = false,
  })  : _tracker = tracker,
        _controller = controller,
        _drawFPS = drawFPS,
        _drawFeatures = drawFeatures,
        _drawFacesRects = drawFacesRects,
        _drawLiveness = drawLiveness,
        super(repaint: tracker);

  @override
  void paint(Canvas canvas, Size size) {
    final int width;
    final int height;

    final extra = Platform.isIOS ? 90 : 0;

    if ((_controller.description.sensorOrientation + extra) % 180 != 0) {
      width = _tracker.height;
      height = _tracker.width;
    } else {
      width = _tracker.width;
      height = _tracker.height;
    }

    // No frame seen yet: nothing to scale against, and dividing by zero here
    // would poison every offset below with infinity.
    if (width <= 0 || height <= 0) {
      _tracker.next();
      return;
    }

    final wc = size.width / width;
    final hc = size.height / height;

    final scale = min(wc, hc);
    final offsetX = (size.width - width * scale) / 2;
    final offsetY = (size.height - height * scale) / 2;

    _faces.clear();

    try {
      for (final face in _tracker.faces()) {
        final position = face.position;

        final tl =
            Offset(position.xc - position.w / 2, position.yc - position.w / 2)
                .scale(scale, scale)
                .translate(offsetX, offsetY);
        final br =
            Offset(position.xc + position.w / 2, position.yc + position.w / 2)
                .scale(scale, scale)
                .translate(offsetX, offsetY);
        FaceRect rect = FaceRect(face.id, tl, br);

        _faces.add(rect);
        if (_drawFacesRects) {
          canvas.drawRect(rect, _paintBlue);
        }

        if (_drawFeatures) {
          for (final point in face.features) {
            canvas.drawCircle(
                Offset(point.x.toDouble(), point.y.toDouble())
                    .scale(scale, scale)
                    .translate(offsetX, offsetY),
                2,
                _paintGreen);
          }
        }

        final name = face.name;
        if (name.isNotEmpty) {
          _textPainter.text = TextSpan(text: name);
          _textPainter.layout();
          _textPainter.paint(
              canvas, rect.bottomCenter.translate(-_textPainter.width / 2, 10));
        }

        if (face.checkLiveness() && _drawLiveness) {
          double liveness = face.liveness;
          bool live = liveness > 0.5;
          String livenessText = "Liveness: ${liveness.toStringAsFixed(3)}";

          _textPainter.text = TextSpan(
              text: livenessText,
              style: TextStyle(
                  color: live ? Colors.lightGreen : Colors.red, fontSize: 14));
          _textPainter.layout();
          _textPainter.paint(canvas, const Offset(10, 30));
        }
      }
    } on FSDK.IdNotFoundError {
      // Tracker was cleared before previous ids were processed
    }

    if (_drawFPS) {
      _textPainter.text = TextSpan(
          text: (1000 / _stopwatch.elapsedMilliseconds).toStringAsFixed(1),
          style: const TextStyle(
              fontSize: 18,
              color: Colors.lightGreen,
              fontWeight: FontWeight.bold));
      _textPainter.layout();
      _textPainter.paint(canvas, const Offset(10, 10));
    }

    _stopwatch.reset();
    _tracker.next();
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) => false;

  FaceRect? getFaceRect(Offset position) {
    for (final rect in _faces) {
      if (rect.contains(position)) {
        return rect;
      }
    }

    return null;
  }
}

import 'dart:async';
import 'dart:developer';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:facial_recognition/src/data/models/recognition_model.dart';
import 'package:facial_recognition/src/data/models/user_model.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:image/image.dart' as img;

typedef _PreprocessParams = ({
  Uint8List bytes,
  int width,
  int height,
  int bytesPerRow,
  int angle,
  bool isIOS,
});

/// Preprocesses camera frame (converts pixel format and rotates).
/// Top-level function so it can run via [compute].
img.Image _preprocessCameraImage(_PreprocessParams p) {
  img.Image image;
  if (p.isIOS) {
    image = img.Image.fromBytes(
      width: p.width,
      height: p.height,
      bytes: p.bytes.buffer,
      rowStride: p.bytesPerRow,
      bytesOffset: p.bytes.offsetInBytes,
      order: img.ChannelOrder.bgra,
      numChannels: 4,
    );
  } else {
    final outImg = img.Image(height: p.height, width: p.width);
    final int frameSize = p.width * p.height;
    final yuv420sp = p.bytes;
    for (int j = 0, yp = 0; j < p.height; j++) {
      int uvp = frameSize + (j >> 1) * p.width, u = 0, v = 0;
      for (int i = 0; i < p.width; i++, yp++) {
        int y = (0xff & yuv420sp[yp]) - 16;
        if (y < 0) y = 0;
        if ((i & 1) == 0) {
          v = (0xff & yuv420sp[uvp++]) - 128;
          u = (0xff & yuv420sp[uvp++]) - 128;
        }
        int y1192 = 1192 * y;
        int r = y1192 + 1634 * v;
        int g = y1192 - 833 * v - 400 * u;
        int b = y1192 + 2066 * u;
        if (r < 0) {
          r = 0;
        } else if (r > 262143) {
          r = 262143;
        }
        if (g < 0) {
          g = 0;
        } else if (g > 262143) {
          g = 262143;
        }
        if (b < 0) {
          b = 0;
        } else if (b > 262143) {
          b = 262143;
        }
        outImg.setPixelRgb(i, j, ((r << 6) & 0xff0000) >> 16,
            ((g >> 2) & 0xff00) >> 8, (b >> 10) & 0xff);
      }
    }
    image = outImg;
  }
  return img.copyRotate(image, angle: p.angle);
}

/// [RecognitionService] class is responsible for performing face recognition on the detected faces.
class RecognitionService {
  /// [threshold] The minimum distance between the face embeddings to be considered as a match. Default is 0.8.
  /// if the distance is less than the threshold, the face is recognized.
  /// decrease the threshold to increase the accuracy of the face recognition.
  final double threshold;

  /// [sensorOrientation] The orientation of the camera sensor.
  final int sensorOrientation;

  /// [users] A list of registered [UserModel] objects.
  final List<UserModel> users;

  /// [RecognitionService] class is responsible for performing face recognition on the detected faces.
  RecognitionService(
      {required this.rotationCompensation,
      required this.sensorOrientation,
      required this.users,
      this.threshold = 0.8}) {
    log('recognition service created');
  }

  /// [rotationCompensation] The rotation compensation to be applied to the image.
  int rotationCompensation;

  /// [isRecognized] A boolean value to check if the face is recognized.
  bool isRecognized = false;

  /// [recognitionModel] An instance of the [RecognitionModel] class.
  RecognitionModel recognitionModel = RecognitionModel();

  /// [recognizedUser] The user that is recognized.
  UserModel? recognizedUser;

  /// Performs face recognition on the provided image frames and detected faces.
  ///
  /// This method processes the provided image frames (either from the camera or a local image) and
  /// detected faces then performs face recognition to identify known users. It updates the
  /// `recognitions` set with the recognized users.
  ///
  /// - Parameters:
  ///   - cameraImageFrame: The [CameraImage] to be processed (if provided).
  ///   - localImageFrame: The [img.Image] to be processed (if provided).
  ///   - faces: A list of detected [Face] objects.
  ///   - recognitions: A set of [UserModel] objects to be updated with recognized users.
  /// - Returns: A boolean value indicating whether any faces were recognized.
  Future<bool> performFaceRecognition({
    CameraImage? cameraImageFrame,
    img.Image? localImageFrame,
    required List<Face> faces,
    required Set<UserModel> recognitions,
  }) async {
    recognitions.clear();
    img.Image? image;

    if (Platform.isAndroid) {
      rotationCompensation = (sensorOrientation + rotationCompensation) % 360;
    }
    final angle = Platform.isIOS ? sensorOrientation : rotationCompensation;
    if (cameraImageFrame != null) {
      // Run image preprocessing in a background thread (doesn't block UI).
      image = await compute(
        _preprocessCameraImage,
        (
          bytes: cameraImageFrame.planes[0].bytes,
          width: cameraImageFrame.width,
          height: cameraImageFrame.height,
          bytesPerRow: cameraImageFrame.planes[0].bytesPerRow,
          angle: angle,
          isIOS: Platform.isIOS,
        ),
      );
      // image = _preprocessCameraImage(
      //   (
      //     bytes: cameraImageFrame.planes[0].bytes,
      //     width: cameraImageFrame.width,
      //     height: cameraImageFrame.height,
      //     bytesPerRow: cameraImageFrame.planes[0].bytesPerRow,
      //     angle: angle,
      //     isIOS: Platform.isIOS,
      //   ),
      // );
    } else if (localImageFrame != null) {
      image = localImageFrame;
    }

    img.Image? croppedFace;

    for (Face face in faces) {
      Rect faceRect = face.boundingBox;
      //crop face
      croppedFace = _cropFaces(image: image!, angle: angle, faceRect: faceRect);

      //pass cropped face to face recognition model
      recognizedUser = recognitionModel.recognize(
          users: users,
          croppedFace: croppedFace,
          location: faceRect,
          face: face);

      log('recognized user: ${recognizedUser?.name}, distance: ${recognizedUser?.distance} < $threshold');
      if (recognizedUser!.distance <= threshold &&
          recognizedUser!.distance >= 0) {
        recognitions.add(recognizedUser!);
        log('Face Recognized !');
        isRecognized = true;
      }
    }

    return isRecognized;
  }

  img.Image _cropFaces({
    required img.Image image,
    required int angle,
    required Rect faceRect,
  }) {
    if (Platform.isAndroid) {
      return img.copyCrop(image,
          x: faceRect.left.toInt(),
          y: faceRect.top.toInt(),
          width: faceRect.width.toInt(),
          height: faceRect.height.toInt());
    }
    int x = faceRect.left.round();
    int y = faceRect.top.round();
    int w = faceRect.width.round();
    int h = faceRect.height.round();

    int cropX, cropY, cropW, cropH;

    switch (angle % 360) {
      case 0:
        cropX = x;
        cropY = y;
        cropW = w;
        cropH = h;
        break;

      case 90:
        cropX = image.width - (y + h);
        cropY = x;
        cropW = h;
        cropH = w;
        break;

      case 180:
        cropX = image.width - (x + w);
        cropY = image.height - (y + h);
        cropW = w;
        cropH = h;
        break;

      case 270:
        cropX = y;
        cropY = image.height - (x + w);
        cropW = h;
        cropH = w;
        break;

      default:
        throw ArgumentError('Angle must be 0, 90, 180, or 270');
    }

    cropX = cropX.clamp(0, image.width - 1);
    cropY = cropY.clamp(0, image.height - 1);
    cropW = cropW.clamp(1, image.width - cropX);
    cropH = cropH.clamp(1, image.height - cropY);

    var cropped = img.copyCrop(
      image,
      x: cropX,
      y: cropY,
      width: cropW,
      height: cropH,
    );

    // Rotate the cropped face back upright for the recognition model.
    switch (angle % 360) {
      case 90:
        cropped = img.copyRotate(cropped, angle: 270);
        break;
      case 180:
        cropped = img.copyRotate(cropped, angle: 180);
        break;
      case 270:
        cropped = img.copyRotate(cropped, angle: 90);
        break;
    }

    return cropped;
  }

  void dispose() {
    recognitionModel.close();
  }
}

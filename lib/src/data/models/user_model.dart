import 'dart:ui';

import 'package:equatable/equatable.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:image/image.dart' as img;

class UserModel extends Equatable {
  final int id;
  final String? image;
  final String name;
  final List<double> embeddings;
  final double distance;
  final img.Image? croppedFace;

  final Rect? location;

  final Face? face;

  /// Constructs a Category.
  const UserModel({
    required this.name,
    required this.id,
    this.image,
    this.face,
    this.location,
    this.croppedFace,
    required this.embeddings,
    required this.distance,
  });

  UserModel copyWith({
    String? name,
    List<double>? embeddings,
    double? distance,
    int? id,
    String? image,
    Rect? location,
    Face? face,
    img.Image? croppedFace,
  }) {
    return UserModel(
      name: name ?? this.name,
      embeddings: embeddings ?? this.embeddings,
      distance: distance ?? this.distance,
      id: id ?? this.id,
      image: image ?? this.image,
      location: location ?? this.location,
      face: face ?? this.face,
      croppedFace: croppedFace ?? this.croppedFace,
    );
  }

  @override
  String toString() {
    return 'UserModel { id: $id, name: $name, image: $image, distance: $distance }';
  }

  @override
  List<Object?> get props => [
        id,
        name,
      ];
}

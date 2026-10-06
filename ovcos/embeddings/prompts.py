"""Fixed instruction views and class templates."""

INSTRUCTIONS = [
    "Represent the object category of the user's input image for classification. Focus on the main object, ignore background.",
    "Represent the object category by focusing on subtle texture irregularities and pattern disruptions that reveal a camouflaged object.",
    "Represent the object category by focusing on boundary cues: outline, contour fragments, and edge discontinuities separating object from background.",
    "Represent the object category using part-based cues. The object may be partially visible or fragmented; focus on distinctive parts rather than full shape.",
    "Represent the object category by emphasizing global shape and pose cues even if color/texture matches the background.",
]

CLASS_TEMPLATES = [
    "a photo of a {class_name}.",
    "a photo of the camouflaged {class_name}.",
    "a photo of the {class_name} camouflaged in the background.",
    "a photo of the {class_name} concealed in the background.",
    "a camouflaged {class_name}.",
    "a {class_name} hiding in the background.",
]

RECOGNITION_INSTRUCTION = (
    "Represent the image for retrieving the most plausible camouflaged object class."
)

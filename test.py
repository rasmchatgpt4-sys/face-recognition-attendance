import cv2, sys
print("cv2.__version__:", cv2.__version__)
print("Has attribute cv2.face ?", hasattr(cv2, "face"))
if hasattr(cv2, "face"):
    print("Has LBPH factory?:", hasattr(cv2.face, "LBPHFaceRecognizer_create"))
else:
    print("cv2.face missing — you likely have opencv-python (no contrib) or a conflicting build.")

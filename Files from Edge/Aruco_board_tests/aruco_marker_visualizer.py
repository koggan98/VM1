import cv2
import numpy as np
from realsense_depth import DepthCamera

# ArUco-Dictionary auswählen
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)

# Parameter für die ArUco-Erkennung
ARUCO_PARAMS = cv2.aruco.DetectorParameters()

# Markergröße in Metern (z. B. 0.08 m für einen Marker mit 8 cm Seitenlänge)
MARKER_SIZE = 0.13


class Aruco:
    def __init__(self, camera):
        self.dc = camera

        # Kameraparameter aus der übergebenen Kamera extrahieren
        rgb_intrinsics = self.dc.rgb_intrinsics
        self.CAMERA_MATRIX = np.array([
            [rgb_intrinsics.fx, 0, rgb_intrinsics.ppx],
            [0, rgb_intrinsics.fy, rgb_intrinsics.ppy],
            [0, 0, 1]
        ], dtype=float)

        self.DIST_COEFFS = np.zeros((5, 1))  # Verzerrungskoeffizienten (standardmäßig keine Verzerrung)

    def run(self):
        """
        Zeigt den Kamerastream und zeichnet erkannte ArUco-Marker mit Koordinatensystem.
        """
        while True:
            # Einzelnes Frame abrufen
            ret, depth_image, color_image = self.dc.get_frame()
            if not ret:
                print("Konnte keine Frames von der RealSense abrufen.")
                break

            # Konvertiere das Bild in Graustufen
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            # ArUco-Marker erkennen
            corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)

            if ids is not None and len(corners) > 0:
                for i in range(len(ids)):
                    # Pose des Markers berechnen
                    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                        corners[i], MARKER_SIZE, self.CAMERA_MATRIX, self.DIST_COEFFS
                    )

                    # Koordinatensystem des Markers zeichnen
                    cv2.drawFrameAxes(color_image, self.CAMERA_MATRIX, self.DIST_COEFFS, rvec, tvec, 0.05)

                    # Markerumrandung und ID zeichnen
                    cv2.aruco.drawDetectedMarkers(color_image, corners, ids)

            # Bild anzeigen
            cv2.imshow("ArUco Marker Detection", color_image)

            # Beenden mit der Taste 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Ressourcen freigeben
        self.dc.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # DepthCamera-Instanz erstellen
    camera = DepthCamera()
    aruco_detector = Aruco(camera)
    aruco_detector.run()
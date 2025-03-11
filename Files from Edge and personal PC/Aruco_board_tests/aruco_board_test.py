import cv2
import numpy as np
from realsense_depth import DepthCamera

# Board-Konfiguration
MARKER_SIZE = 0.08  # Größe eines Markers in Metern (8 cm)
MARKER_SEPARATION = 0.02  # Abstand zwischen den Markern in Metern (2 cm)
BOARD_ROWS = 2
BOARD_COLS = 2
MARKER_IDS = [100, 105, 110, 115]  # Benutzerdefinierte Marker-IDs

# ArUco-Dictionary auswählen
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)

# Parameter für die ArUco-Erkennung
ARUCO_PARAMS = cv2.aruco.DetectorParameters()

class ArucoBoard:
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

        self.board = cv2.aruco.Board(
            objPoints=[self._create_marker_corners(i) for i in MARKER_IDS],
            dictionary=ARUCO_DICT,
            ids=np.array(MARKER_IDS, dtype=np.int32).reshape(-1, 1)
        )

    def _create_marker_corners(self, marker_id):
        """Erstellt die Eckpunkte für einen einzelnen Marker im Board."""
        half_size = MARKER_SIZE / 2
        return np.array([
            [-half_size, half_size, 0],
            [half_size, half_size, 0],
            [half_size, -half_size, 0],
            [-half_size, -half_size, 0]
        ], dtype=np.float32)

    def run(self):
        """Zeigt den Kamerastream und zeichnet das erkannte ArUco-Board mit Koordinatensystem."""
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
                # Initialisiere rvec und tvec als leere Arrays
                rvec = np.zeros((1, 3), dtype=np.float64)
                tvec = np.zeros((1, 3), dtype=np.float64)

                # Pose des ArUco-Boards berechnen
                ret, rvec, tvec = cv2.aruco.estimatePoseBoard(corners, ids, self.board, self.CAMERA_MATRIX, self.DIST_COEFFS, rvec, tvec)

                if ret > 0:
                    # Koordinatensystem des Boards zeichnen
                    cv2.drawFrameAxes(color_image, self.CAMERA_MATRIX, self.DIST_COEFFS, rvec, tvec, 0.05)

                    # Board-Markierungen und IDs zeichnen
                    cv2.aruco.drawDetectedMarkers(color_image, corners, ids)

                    # Pose-Informationen ausgeben
                    print(f"Board erkannt! tvec: {tvec.flatten()}, rvec: {rvec.flatten()}")

            # Bild anzeigen
            cv2.imshow("ArUco Board Detection", color_image)

            # Beenden mit der Taste 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Ressourcen freigeben
        self.dc.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # DepthCamera-Instanz erstellen
    camera = DepthCamera()
    aruco_board_detector = ArucoBoard(camera)
    aruco_board_detector.run()

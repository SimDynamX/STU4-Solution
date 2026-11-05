"""
camera_overlay.py - project radar contacts into a camera image (a calibration /
debugging aid), using the camera model from the wiki's Vehicle Specs page.

Wiki model: the body-to-camera matrix is  T = Ry(pitch) @ Rz(yaw)  (3-2-1 Euler,
zero roll). Rows of T are the camera axes written in body coordinates:
    row 0 = boresight (camera +x), row 1 = camera y (points LEFT), row 2 = camera z (UP).
Positive yaw turns the camera left, positive pitch turns it DOWN.

Assumptions you should verify with a real image (the wiki doesn't say):
  * the image is a pinhole view looking down camera +x, with image-right = -y and
    image-up = +z;
  * `fov_deg` is the horizontal field of view (identical to vertical for the
    square images the starter code captures).
If circles come out mirrored or shifted, flip FLIP_U / FLIP_V below.
"""

import numpy as np
from PIL import ImageDraw

CAM_POS_BODY = {  # metres, shuttle body frame (wiki)
    "top": np.array([1.27, 0.0, 0.9]),
    "bottom": np.array([1.675, 0.0, -1.025]),
}
FLIP_U = False
FLIP_V = False


def body_to_camera(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    psi, th = np.deg2rad(yaw_deg), np.deg2rad(pitch_deg)
    Ry = np.array([[np.cos(th), 0, -np.sin(th)], [0, 1, 0], [np.sin(th), 0, np.cos(th)]])
    Rz = np.array([[np.cos(psi), np.sin(psi), 0], [-np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
    return Ry @ Rz


def project(p_body, camera: str, yaw_deg: float, pitch_deg: float, fov_deg: float, size_px: int):
    """
    p_body: a point relative to the shuttle centre of mass, in the BODY frame.
    Returns (u, v, f_px, depth) in pixels (f_px = focal length in pixels,
    depth = distance along the boresight), or None if it is behind the camera.
    """
    v_cam = body_to_camera(yaw_deg, pitch_deg) @ (np.asarray(p_body, float) - CAM_POS_BODY[camera])
    if v_cam[0] <= 0.1:
        return None
    f = (size_px / 2) / np.tan(np.deg2rad(fov_deg) / 2)
    u = size_px / 2 - f * v_cam[1] / v_cam[0]  # camera y points left -> subtract
    v = size_px / 2 - f * v_cam[2] / v_cam[0]  # camera z points up   -> subtract
    if FLIP_U:
        u = size_px - u
    if FLIP_V:
        v = size_px - v
    return u, v, f, v_cam[0]


def overlay_contacts(image, contacts, camera, yaw_deg, pitch_deg, fov_deg):
    """
    image    : square PIL image from the camera
    contacts : list of (p_body, diameter_m); p_body = radar centre rotated into the body frame
    Returns a copy with a circle drawn for every contact inside the view.
    """
    out = image.copy()
    draw = ImageDraw.Draw(out)
    n = out.size[0]
    for p_body, diam in contacts:
        r = project(p_body, camera, yaw_deg, pitch_deg, fov_deg, n)
        if r is None:
            continue
        u, v, f, depth = r
        rad = max(f * (diam / 2) / depth, 2.0)
        if -rad < u < n + rad and -rad < v < n + rad:
            draw.ellipse([u - rad, v - rad, u + rad, v + rad], outline=(255, 60, 60), width=2)
            draw.text((u + rad + 2, v - rad), f"{depth:.0f}m", fill=(255, 220, 0))
    return out


# ------------------------------------------------------------------------------
# HOW TO USE IN Controller.py (inside the existing "if image is not None:" block)
#
#   from camera_overlay import overlay_contacts          # near the other imports
#   ...
#   contacts = [(to_body(d[0, :]), d[1, 2]) for d in GetNearestDebrisLocations(100)]
#   marked = overlay_contacts(imageRGB, contacts, "top",
#                             yaw_deg=30.0, pitch_deg=-10.0,   # same numbers you gave PanTiltCamera
#                             fov_deg=cameraParams.FOV)
#   marked.save(os.path.join(
#       os.path.dirname(os.path.abspath(__file__)), f"cam_{imageNumber:04d}.png"))
# ------------------------------------------------------------------------------

import socket
import json


# ------------------------------------------------------------------
# Unit convention:
#   The real FEI/TFS AutoScript API uses SI units (meters, radians).
#   Your simulator uses nanometers internally.
#   => convert meters -> nm by multiplying by 1e9.
# ------------------------------------------------------------------
_M_TO_NM = 1e9

class MagnificationData:
    """
    Mimics autoscript_tem_microscope_client.structures.MagnificationData.

    Contains properties of a TEM magnification.
    """

    def __init__(self, nominal, fov_nm):
        self._nominal = nominal
        self._fov_nm = fov_nm  # internal: corresponding FOV in nm

    @property
    def nominal(self):
        """The nominal magnification of the projector."""
        return self._nominal

    @property
    def calibrated(self):
        """The calibrated magnification of the projector."""
        # For this simulator, calibrated == nominal
        return self._nominal

    @property
    def rotation(self):
        """The rotation of the magnification (radians)."""
        return 0.0

    @property
    def label(self):
        """A label fully describing the magnification."""
        return f"{self._nominal:.0f}x"

    @property
    def magnification_sub_mode(self):
        """The magnification sub-mode (None for LM)."""
        return None

    @property
    def objective_lens_mode(self):
        """The objective lens mode."""
        return "LM"

    def __repr__(self):
        return f"MagnificationData(nominal={self._nominal}, label='{self.label}')"

    def __eq__(self, other):
        if isinstance(other, MagnificationData):
            return self._nominal == other._nominal
        return False

    def __hash__(self):
        return hash(self._nominal)

class StagePosition:
    """
    Mimics autoscript_tem_microscope_client.structures.StagePosition.

    Axes: x, y, z in meters; a (alpha), b (beta) in radians.
    Any axis left as None is "not specified" and will not be moved.
    """

    def __init__(self, x=None, y=None, z=None, a=None, b=None):
        self.x = x
        self.y = y
        self.z = z
        self.a = a
        self.b = b

    @classmethod
    def _from_sequence(cls, seq):
        """Build a StagePosition from a list/tuple: [x, y, z, a, b]."""
        seq = list(seq)
        if len(seq) > 5:
            raise ValueError("StagePosition sequence can have at most 5 elements (x, y, z, a, b).")
        vals = seq + [None] * (5 - len(seq))
        return cls(*vals)

    def __repr__(self):
        return (f"StagePosition(x={self.x}, y={self.y}, z={self.z}, "
                f"a={self.a}, b={self.b})")

class StageVelocity:
    """
    Mimics autoscript_tem_microscope_client.structures.StageVelocity.

    Axes: x, y, z in m/s; a (alpha), b (beta) in rad/s.
    Any axis left as None is "not specified".
    """

    def __init__(self, x=None, y=None, z=None, a=None, b=None):
        self.x = x
        self.y = y
        self.z = z
        self.a = a
        self.b = b

    @classmethod
    def _from_sequence(cls, seq):
        seq = list(seq)
        if len(seq) > 5:
            raise ValueError("StageVelocity sequence can have at most 5 elements (x, y, z, a, b).")
        vals = seq + [None] * (5 - len(seq))
        return cls(*vals)

    def __repr__(self):
        return (f"StageVelocity(x={self.x}, y={self.y}, z={self.z}, "
                f"a={self.a}, b={self.b})")
    

class _StageServerProxy:
    """
    Handles the actual TCP communication with the simulator server.
    """

    def __init__(self, host, port):
        self._host = host
        self._port = port

    def send_cmd(self, cmd, args=None):
        if args is None:
            args = {}
        msg = {"cmd": cmd, "args": args}
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self._host, self._port))
            s.sendall(json.dumps(msg).encode())
            resp = s.recv(4096).decode()
        return json.loads(resp)
    def wait_for_stage(self, poll_interval=0.05):
        """Block until the stage reports it is no longer moving."""
        import time
        # Small initial delay so the server has registered the move
        # before we start polling (avoids a race where is_moving is
        # still False on the very first check).
        time.sleep(poll_interval)
        while self.send_cmd("is_moving")["values"]["moving"]:
            time.sleep(poll_interval)

class Magnification:
    """
    Mimics microscope.optics.magnification.

    Provides access to the TEM magnifications of the microscope.
    """

    # Reference: FOV=1e6 nm corresponds to 1000x magnification.
    # magnification = _REF_MAG * _REF_FOV / fov
    _REF_FOV = 1e6   # nm
    _REF_MAG = 1000  # x

    def __init__(self, proxy):
        self._proxy = proxy
        # Build the list of available magnifications from the simulator's zoom_levels.
        # The simulator exposes zoom_levels via stageInfo... but actually it doesn't.
        # So we hardcode the same array here (must match Microscopy.zoom_levels).
        self._zoom_levels_nm = [
            m * 10**e for e in range(3, 6) for m in range(1, 10)
        ] + [1e6]
        self._zoom_levels_nm = sorted(self._zoom_levels_nm)

        # Pre-build MagnificationData for each FOV (smaller FOV = higher mag).
        self._mag_data_by_fov = {}
        for fov in self._zoom_levels_nm:
            mag = self._fov_to_mag(fov)
            self._mag_data_by_fov[fov] = MagnificationData(nominal=mag, fov_nm=fov)

    def _fov_to_mag(self, fov_nm):
        """Convert FOV (nm) to magnification value."""
        return self._REF_MAG * self._REF_FOV / fov_nm

    def _mag_to_fov(self, mag):
        """Convert magnification to FOV (nm)."""
        return self._REF_MAG * self._REF_FOV / mag

    @property
    def available_values(self):
        """
        Returns the full collection of magnifications for the current optical state.

        Returns
        -------
        List[MagnificationData]
            Sorted from lowest to highest magnification.
        """
        # Sorted by magnification ascending (which is FOV descending).
        fovs_desc = sorted(self._zoom_levels_nm, reverse=True)
        return [self._mag_data_by_fov[fov] for fov in fovs_desc]

    @property
    def value(self):
        """
        The current magnification.

        Returns
        -------
        MagnificationData
        """
        info = self._proxy.send_cmd("stageInfo")["values"]
        current_fov = info["FOV"]
        # Find the closest matching FOV in our list.
        closest_fov = min(self._zoom_levels_nm, key=lambda f: abs(f - current_fov))
        return self._mag_data_by_fov[closest_fov]

    @value.setter
    def value(self, mag_data):
        """
        Set the current magnification.

        Parameters
        ----------
        mag_data : MagnificationData
            Must be one of the values from available_values.
        """
        if not isinstance(mag_data, MagnificationData):
            raise TypeError("value must be a MagnificationData from available_values.")

        # Validate it's in our available set.
        if mag_data._fov_nm not in self._mag_data_by_fov:
            raise ValueError("Invalid magnification; must be from available_values.")

        target_fov = mag_data._fov_nm
        current_fov = self._proxy.send_cmd("stageInfo")["values"]["FOV"]

        # Compute how many steps to reach target_fov.
        # The server's `increase(steps)` moves toward smaller FOV (higher mag) for positive steps.
        current_idx = self._zoom_levels_nm.index(
            min(self._zoom_levels_nm, key=lambda f: abs(f - current_fov))
        )
        target_idx = self._zoom_levels_nm.index(target_fov)
        steps = current_idx - target_idx  # positive = zoom in (smaller FOV)

        if steps != 0:
            self._proxy.send_cmd("increase", {"steps": steps})

    def increase(self, steps):
        """
        Increases the magnification by steps (may be negative, clipped if necessary).

        Parameters
        ----------
        steps : int
            Positive = increase magnification (zoom in).
            Negative = decrease magnification (zoom out).

        Returns
        -------
        MagnificationData
            The new magnification after the change.
        """
        steps = int(steps)
        self._proxy.send_cmd("increase", {"steps": steps})
        return self.value
    


class Stage:
    """
    Mimics microscope.specimen.stage.
    """

    def __init__(self, proxy):
        self._proxy = proxy
        # Current jog velocity vector in nm/s (simulator units).
        self._jog_vx_nm = 0.0
        self._jog_vy_nm = 0.0

    def _ensure_not_jogging(self):
        """Raise if a jog is active, mirroring the real instrument."""
        if (self._jog_vx_nm != 0.0) or (self._jog_vy_nm != 0.0):
            raise RuntimeError(
                "Stage not ready: the stage is currently jogging. "
                "Call stop_jogging() before issuing a move."
            )
    def absolute_move(self, position):
        """
        Moves the stage to a new absolute position. Blocks until the move
        has finished.

        Parameters
        ----------
        position : StagePosition | list | tuple
            The absolute position to which the stage should move.
            Axes with value None are not affected by the move.

        Notes
        -----
        Only the X and Y axes are supported by this simulator. Z / alpha /
        beta values are accepted but ignored (the simulator has no such axes).
        """
        self._ensure_not_jogging()

        if isinstance(position, (list, tuple)):
            position = StagePosition._from_sequence(position)
        elif not isinstance(position, StagePosition):
            raise TypeError("position must be a StagePosition, list, or tuple.")

        info = self._proxy.send_cmd("stageInfo")["values"]
        cur_x_nm = info["x"]
        cur_y_nm = info["y"]

        target_x_nm = cur_x_nm if position.x is None else position.x * _M_TO_NM
        target_y_nm = cur_y_nm if position.y is None else position.y * _M_TO_NM

        self._proxy.send_cmd("absolute_move", {"x": target_x_nm, "y": target_y_nm})
        self._proxy.wait_for_stage()

    def relative_move(self, relative_position):
        """
        Moves the stage relatively to the current stage position. Blocks
        until the move has finished.

        Parameters
        ----------
        relative_position : StagePosition | list | tuple
            The delta position by which the stage should move.
            Axes with value None are not affected by the move.

        Notes
        -----
        Only the X and Y axes are supported by this simulator. Z / alpha /
        beta deltas are accepted but ignored (the simulator has no such axes).
        """
        self._ensure_not_jogging()

        if isinstance(relative_position, (list, tuple)):
            relative_position = StagePosition._from_sequence(relative_position)
        elif not isinstance(relative_position, StagePosition):
            raise TypeError(
                "relative_position must be a StagePosition, list, or tuple."
            )

        dx_nm = 0.0 if relative_position.x is None else relative_position.x * _M_TO_NM
        dy_nm = 0.0 if relative_position.y is None else relative_position.y * _M_TO_NM

        self._proxy.send_cmd("relative_move", {"dx": dx_nm, "dy": dy_nm})
        self._proxy.wait_for_stage()

    @property
    def position(self):
        """
        Current stage position as a StagePosition (SI units: meters/radians).

        The simulator only has X and Y axes; Z, alpha and beta are reported
        as 0.0 to match the shape of the real API's StagePosition.
        """
        info = self._proxy.send_cmd("stageInfo")["values"]
        return StagePosition(
            x=info["x"] / _M_TO_NM,
            y=info["y"] / _M_TO_NM,
            z=0.0,
            a=0.0,
            b=0.0,
        ) 
    
    def start_jogging(self, jogging_velocity):
        """
        Starts jogging at a given velocity. Asynchronous: returns immediately.

        Parameters
        ----------
        jogging_velocity : StageVelocity | list | tuple
            The velocity vector of the stage movement. Axes with value None
            keep their current jog velocity; specifying an axis again overrides
            its previous value.

        Notes
        -----
        Only X and Y are supported by this simulator; Z / alpha / beta
        velocities are accepted but ignored.
        """
        import math

        if isinstance(jogging_velocity, (list, tuple)):
            jogging_velocity = StageVelocity._from_sequence(jogging_velocity)
        elif not isinstance(jogging_velocity, StageVelocity):
            raise TypeError(
                "jogging_velocity must be a StageVelocity, list, or tuple."
            )

        # Update only the specified axes (None => keep current), convert m/s -> nm/s.
        if jogging_velocity.x is not None:
            self._jog_vx_nm = jogging_velocity.x * _M_TO_NM
        if jogging_velocity.y is not None:
            self._jog_vy_nm = jogging_velocity.y * _M_TO_NM

        # Convert the combined vector to (speed, angle) for the drift command.
        speed = math.hypot(self._jog_vx_nm, self._jog_vy_nm)
        angle = math.atan2(self._jog_vy_nm, self._jog_vx_nm)

        self._proxy.send_cmd("drift", {"velocity": speed, "angle": angle})

    def stop_jogging(self):
        """
        Stops jogging on all axes.
        """
        self._jog_vx_nm = 0.0
        self._jog_vy_nm = 0.0
        self._proxy.send_cmd("stop_drift")

    @property
    def is_moving(self):
        """
        True if the stage is currently moving (targeted move or jogging).
        """
        moving = self._proxy.send_cmd("is_moving")["values"]["moving"]
        jogging = (self._jog_vx_nm != 0.0) or (self._jog_vy_nm != 0.0)
        return bool(moving or jogging)
        
class Specimen:
    """
    Mimics microscope.specimen.
    """

    def __init__(self, proxy):
        self.stage = Stage(proxy)

class Optics:
    """
    Mimics microscope.optics.
    """

    def __init__(self, proxy):
        self.magnification = Magnification(proxy)

class TemMicroscopeClient:
    """
    Doppelganger of autoscript_tem_microscope_client.TemMicroscopeClient.

    Usage
    -----
    microscope = TemMicroscopeClient()
    microscope.connect()
    microscope.specimen.stage.absolute_move(StagePosition(x=250e-6))
    """

    def __init__(self):
        self._proxy = None
        self.specimen = None

    def connect(self, host="127.0.0.1", port=9999):
        """
        Build the connection to the (digital) microscope.
        """
        self._proxy = _StageServerProxy(host, port)
        self._proxy.send_cmd("stageInfo")
        self.specimen = Specimen(self._proxy)
        self.optics = Optics(self._proxy)
        print(f"Connected to microscope at {host}:{port}")
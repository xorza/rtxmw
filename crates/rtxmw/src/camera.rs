//! Free-flying noclip camera in Morrowind world space.

use glam::{Mat4, Vec3};

/// Morrowind world units per metre — the game uses 64 units per yard.
///
/// OpenMW carries this as `69.99125109`; that is beyond `f32` precision and rounds to this literal.
const UNITS_PER_METRE: f32 = 69.991_25;

/// Pitch is clamped just short of vertical so the view basis never degenerates.
const PITCH_LIMIT: f32 = std::f32::consts::FRAC_PI_2 - 0.001;

/// A Z-up right-handed fly camera: +X east, +Y north, +Z up, matching Morrowind's convention.
#[derive(Debug)]
pub struct Camera {
    position: Vec3,
    /// Rotation about +Z. Zero looks along +X (east).
    yaw: f32,
    /// Rotation above the horizon, clamped to just short of straight up or down.
    pitch: f32,
    /// Read only by [`Camera::projection`], which nothing calls until there are rays to generate.
    #[allow(dead_code)]
    fov_y: f32,
    /// Metres per second; converted to world units on use.
    speed: f32,
}

/// Which way the camera is being asked to move this frame, in camera-relative axes.
#[derive(Debug, Clone, Copy)]
pub struct Movement {
    pub forward: f32,
    pub right: f32,
    pub up: f32,
    /// Multiplier applied to [`Camera::speed`], for a sprint modifier.
    pub boost: f32,
}

// Derived `Default` would leave `boost` at zero, which would mean "stand still" rather than
// "no sprint" — a silent trap for anyone writing `Movement { forward: 1.0, ..default() }`.
impl Default for Movement {
    fn default() -> Self {
        Self {
            forward: 0.0,
            right: 0.0,
            up: 0.0,
            boost: 1.0,
        }
    }
}

impl Camera {
    /// A camera at `position` looking east along +X.
    pub fn new(position: Vec3) -> Self {
        Self {
            position,
            yaw: 0.0,
            pitch: 0.0,
            fov_y: 75f32.to_radians(),
            speed: 8.0,
        }
    }

    /// Applies mouse look. Deltas are in pixels; the scale is radians per pixel.
    pub fn look(&mut self, delta_x: f32, delta_y: f32) {
        const RADIANS_PER_PIXEL: f32 = 0.0022;
        self.yaw -= delta_x * RADIANS_PER_PIXEL;
        self.pitch = (self.pitch - delta_y * RADIANS_PER_PIXEL).clamp(-PITCH_LIMIT, PITCH_LIMIT);
    }

    /// Integrates `movement` over `dt` seconds.
    pub fn fly(&mut self, movement: Movement, dt: f32) {
        let distance = self.speed * movement.boost * UNITS_PER_METRE * dt;

        let forward = self.forward();
        // Horizontal strafe: crossing forward with world up keeps movement level regardless of pitch.
        let right = forward.cross(Vec3::Z).normalize_or_zero();

        let direction = forward * movement.forward + right * movement.right + Vec3::Z * movement.up;
        self.position += direction.normalize_or_zero() * distance;
    }

    /// Unit vector the camera is looking along.
    pub fn forward(&self) -> Vec3 {
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        Vec3::new(cos_pitch * cos_yaw, cos_pitch * sin_yaw, sin_pitch)
    }

    /// Right-handed view matrix.
    ///
    /// Unused until ray generation exists at M3; kept because its convention is pinned by tests and
    /// getting it wrong later is expensive to debug.
    #[allow(dead_code)]
    pub fn view(&self) -> Mat4 {
        glam::camera::rh::view::look_to_mat4(self.position, self.forward(), Vec3::Z)
    }

    /// Reverse-Z infinite perspective projection in Vulkan NDC — Z in [0, 1], Y-down.
    ///
    /// Reverse-Z distributes float depth precision far better than a conventional near/far mapping
    /// and costs nothing to adopt now, while retrofitting it later would touch every depth compare.
    #[allow(dead_code)]
    pub fn projection(&self, aspect: f32) -> Mat4 {
        glam::camera::rh::proj::vulkan::perspective_infinite_reverse(
            self.fov_y,
            aspect,
            self.near_plane(),
        )
    }

    /// Near plane in world units — 5 cm, well inside anything the player can approach.
    #[allow(dead_code)]
    pub fn near_plane(&self) -> f32 {
        0.05 * UNITS_PER_METRE
    }

    /// Current position in world units.
    pub fn position(&self) -> Vec3 {
        self.position
    }

    /// Movement speed in metres per second.
    pub fn speed(&self) -> f32 {
        self.speed
    }

    /// Scales the movement speed, clamped to a usable range.
    pub fn scale_speed(&mut self, factor: f32) {
        self.speed = (self.speed * factor).clamp(0.5, 2000.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_is_east_at_rest_and_turns_left_with_positive_yaw() {
        let camera = Camera::new(Vec3::ZERO);
        // Yaw 0, pitch 0 looks along +X.
        assert!((camera.forward() - Vec3::X).length() < 1e-6);

        // A quarter turn of positive yaw about +Z takes east to north (+Y).
        let mut turned = Camera::new(Vec3::ZERO);
        turned.yaw = std::f32::consts::FRAC_PI_2;
        assert!((turned.forward() - Vec3::Y).length() < 1e-6);
    }

    #[test]
    fn pitch_clamps_short_of_vertical() {
        let mut camera = Camera::new(Vec3::ZERO);
        camera.look(0.0, -100_000.0);
        assert!(camera.pitch < PITCH_LIMIT + 1e-6);
        assert!(camera.pitch > PITCH_LIMIT - 1e-3);
        // Still a usable basis: forward is not parallel to world up.
        assert!(camera.forward().cross(Vec3::Z).length() > 1e-3);
    }

    #[test]
    fn flying_forward_one_second_covers_speed_times_units_per_metre() {
        let mut camera = Camera::new(Vec3::ZERO);
        let movement = Movement {
            forward: 1.0,
            ..Movement::default()
        };
        camera.fly(movement, 1.0);
        // 8 m/s * 69.99125109 units/m * 1 s = 559.93 units, due east.
        let expected = 8.0 * UNITS_PER_METRE;
        assert!((camera.position().x - expected).abs() < 1e-2);
        assert!(camera.position().y.abs() < 1e-6);
        assert!(camera.position().z.abs() < 1e-6);
    }

    #[test]
    fn strafing_stays_level_even_when_pitched() {
        let mut camera = Camera::new(Vec3::ZERO);
        camera.pitch = 1.0;
        let movement = Movement {
            right: 1.0,
            ..Movement::default()
        };
        camera.fly(movement, 1.0);
        // Looking up should not tilt strafe out of the XY plane.
        assert!(camera.position().z.abs() < 1e-6);
    }

    #[test]
    fn diagonal_movement_is_not_faster_than_straight() {
        let mut straight = Camera::new(Vec3::ZERO);
        straight.fly(
            Movement {
                forward: 1.0,
                ..Movement::default()
            },
            1.0,
        );

        let mut diagonal = Camera::new(Vec3::ZERO);
        diagonal.fly(
            Movement {
                forward: 1.0,
                right: 1.0,
                ..Movement::default()
            },
            1.0,
        );

        let straight_distance = straight.position().length();
        let diagonal_distance = diagonal.position().length();
        assert!((straight_distance - diagonal_distance).abs() < 1e-2);
    }

    #[test]
    fn reverse_z_projection_maps_near_to_one_and_infinity_to_zero() {
        let camera = Camera::new(Vec3::ZERO);
        let projection = camera.projection(16.0 / 9.0);

        // A point on the near plane, straight ahead in view space (which looks down -Z).
        let near = projection * glam::Vec4::new(0.0, 0.0, -camera.near_plane(), 1.0);
        assert!((near.z / near.w - 1.0).abs() < 1e-4);

        let far = projection * glam::Vec4::new(0.0, 0.0, -1.0e9, 1.0);
        assert!((far.z / far.w).abs() < 1e-4);
    }

    #[test]
    fn speed_scaling_clamps_at_both_ends() {
        let mut camera = Camera::new(Vec3::ZERO);
        for _ in 0..100 {
            camera.scale_speed(0.5);
        }
        assert!((camera.speed() - 0.5).abs() < 1e-6);

        for _ in 0..100 {
            camera.scale_speed(2.0);
        }
        assert!((camera.speed() - 2000.0).abs() < 1e-3);
    }
}

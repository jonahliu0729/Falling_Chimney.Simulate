import numpy as np
import matplotlib.pyplot as plt

# ---------------- Parameters ----------------
g = 9.81
L = 50
break_strength = 20000
theta0 = np.deg2rad(5)
dt = 0.01
max_time = 8

frames_bottom = []     # list of segments: [(x1,y1,x2,y2,is_bottom_before_break_bool), ...]
frames_top = []        # list of top-piece segments over time (x1,y1,x2,y2)
broke = False

theta = theta0
omega = 0.0
t = 0.0

# ---------- helpers ----------
def rod_endpoints(length, angle, base=(0.0, 0.0)):
    """Return endpoints (x1,y1,x2,y2) for a rod of given length and angle from vertical."""
    x1, y1 = base
    x2 = x1 + length*np.sin(angle)
    y2 = y1 + length*np.cos(angle)
    return x1, y1, x2, y2

def top_piece_endpoints(center_x, center_y, angle, length):
    """Return endpoints of the top piece, given its center and angle."""
    dx = (length/2)*np.sin(angle)
    dy = (length/2)*np.cos(angle)
    x1, y1 = center_x - dx, center_y - dy
    x2, y2 = center_x + dx, center_y + dy
    return x1, y1, x2, y2

def clip_to_ground(x1, y1, x2, y2):
    """
    Clip the segment to y>=0. If both endpoints are below ground, return None.
    If one endpoint is below, move it to the intersection with y=0.
    """
    # Both above or on ground: nothing to do
    if y1 >= 0 and y2 >= 0:
        return x1, y1, x2, y2
    # Both below: drop segment
    if y1 <= 0 and y2 <= 0:
        return None
    # One above, one below: find intersection with y=0
    # Parametric: P(t) = P1 + t*(P2-P1), find t s.t. y=0
    if y2 != y1:
        t = (0 - y1) / (y2 - y1)
    else:
        return None  # horizontal below ground; should not happen in our use
    xi = x1 + t*(x2 - x1)
    yi = 0.0
    if y1 < 0:   # move endpoint 1 up to ground
        return xi, yi, x2, y2
    else:        # move endpoint 2 down to ground
        return x1, y1, xi, yi

# ---------------- Simulation ----------------
while t < max_time:
    alpha = (3 * g / (2 * L)) * np.sin(theta)
    # simple “bending force” proxy just to trigger a break
    bending_force = abs(np.sin(theta) * (L - L/3)) * 1000

    # BREAK EVENT
    if not broke and bending_force > break_strength:
        broke = True
        break_point = L * 0.33
        bottom_len = break_point
        top_len = L - break_point

        # top piece state at break
        top_center_dist = break_point + top_len/2
        top_x = top_center_dist * np.sin(theta)
        top_y = top_center_dist * np.cos(theta)
        v_x = omega * top_center_dist * np.cos(theta)
        v_y = -omega * top_center_dist * np.sin(theta)
        top_angle = theta
        top_omega = omega

        print(f"Break at t={t:.2f}s, {break_point:.1f} m from base")

    # BEFORE BREAK: whole chimney rotates
    if not broke:
        # advance dynamics
        omega += alpha * dt
        theta += omega * dt

        # store one segment of whole rod (will be clipped at draw time)
        x1,y1,x2,y2 = rod_endpoints(L, theta)
        frames_bottom.append((x1,y1,x2,y2, True))

    # AFTER BREAK: bottom rotates; top flies & spins
    else:
        # --- bottom piece dynamics ---
        bottom_alpha = (3 * g / (2 * bottom_len)) * np.sin(theta)
        omega += bottom_alpha * dt
        theta += omega * dt
        x1,y1,x2,y2 = rod_endpoints(bottom_len, theta)
        frames_bottom.append((x1,y1,x2,y2, False))

        # --- top piece projectile + spin ---
        v_y -= g * dt
        top_x += v_x * dt
        top_y += v_y * dt
        top_angle += top_omega * dt

        # compute endpoints
        x1t,y1t,x2t,y2t = top_piece_endpoints(top_x, top_y, top_angle, top_len)

        # COLLISION CHECK BEFORE SAVING: stop exactly when any end hits ground
        if y1t <= 0 or y2t <= 0:
            # clip the last segment to lie on/above ground so nothing dips under
            clipped = clip_to_ground(x1t,y1t,x2t,y2t)
            if clipped is not None:
                frames_top.append(clipped)
            break
        else:
            frames_top.append((x1t,y1t,x2t,y2t))

    t += dt

# ---------------- Plot ----------------
plt.figure(figsize=(8,8))
plt.axis('equal')
plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")
plt.title("Falling Chimney (Clipped at Ground, No Lines Below y=0)")

# draw ground
plt.plot([-10, L+10], [0, 0])

# draw bottom piece history (sparse)
for (x1,y1,x2,y2, prebreak) in frames_bottom[::10]:
    seg = clip_to_ground(x1,y1,x2,y2)
    if seg is None:
        continue
    x1c,y1c,x2c,y2c = seg
    plt.plot([x1c,x2c],[y1c,y2c], 'b' if prebreak else 'r')

# draw top piece history (sparse)
for (x1,y1,x2,y2) in frames_top[::5]:
    seg = clip_to_ground(x1,y1,x2,y2)
    if seg is None:
        continue
    x1c,y1c,x2c,y2c = seg
    plt.plot([x1c,x2c],[y1c,y2c], color='orange')

plt.show()

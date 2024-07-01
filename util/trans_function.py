import numpy as np

# def RGB2HSV(RGB):
#     R, G, B = RGB[:,:,0], RGB[:,:,1], RGB[:,:,2]
#     maxc = np.maximum(R,G,B)
#     minc = np.minimum(R,G,B)
#     V = maxc
#     deltac = maxc - minc
#     S = np.where(maxc == 0, 0, deltac/maxc)
#     rc = ((G - B) / deltac) % 6.0
#     gc = (B - R) / deltac + 2.0
#     bc = (R - G) / deltac + 4.0
#     H = np.where(maxc == minc, 0, np.where(maxc == R, rc, np.where(maxc == G, gc, bc)))
#     H = (H/6.0)
#     HSV = np.stack([H,S,V], axis=-1)
#     return HSV

def RGB2HSV(RGB):
    RGB = RGB / 255.0
    R, G, B = RGB[:,:,0], RGB[:,:,1], RGB[:,:,2]
    maxc = np.maximum(R,G,B)
    minc = np.minimum(R,G,B)
    V = maxc
    deltac = maxc - minc
    S = np.where(maxc == 0, 0, deltac/maxc)
    rc = (60 * (G - B) / deltac + 360) % 360
    gc = (60 * (B - R) / deltac + 120) % 360
    bc = (60 * (R - G) / deltac + 240) % 360
    H = np.where(maxc == minc, 0, np.where(maxc == R, rc, np.where(maxc == G, gc, bc)))
    # H = (H/6.0)
    HSV = np.stack([H,S,V], axis=-1)
    return HSV

def HSV2RGB(HSV):
    H, S, V = HSV[:,:,0], HSV[:,:,1], HSV[:,:,2]
    i = (H * 6.0).astype(int)
    f = (H * 6.0) - i
    p = V * (1.0 - S)
    q = V * (1.0 - S * f)
    t = V * (1.0 - S * (1.0 - f))
    i = i % 6
    RGB = np.zeros_like(HSV)
    idx = i == 0
    RGB[idx] = np.transpose([V[idx], t[idx], p[idx]])
    idx = i == 1
    RGB[idx] = np.transpose([q[idx], V[idx], p[idx]])
    idx = i == 2
    RGB[idx] = np.transpose([p[idx], V[idx], t[idx]])
    idx = i == 3
    RGB[idx] = np.transpose([p[idx], q[idx], V[idx]])
    idx = i == 4
    RGB[idx] = np.transpose([t[idx], p[idx], V[idx]])
    idx = i == 5
    RGB[idx] = np.transpose([V[idx], p[idx], q[idx]])
    return RGB

# def HSV2RGB(HSV):
#     H, S, V = HSV[:, :, 0], HSV[:, :, 1], HSV[:, :, 2]
#     if S == 0:  # 灰色，饱和度为0
#         R = G = B = V
#         return R, G, B
#     H /= 60  # sector 0 to 5
#     i = int(H)
#     f = H - i  # factorial part of H
#     p = V * (1 - S)
#     q = V * (1 - S * f)
#     t = V * (1 - S * (1 - f))
#     if i == 0:
#         R, G, B = V, t, p
#     elif i == 1:
#         R, G, B = q, V, p
#     elif i == 2:
#         R, G, B = p, V, t
#     elif i == 3:
#         R, G, B = p, q, V
#     elif i == 4:
#         R, G, B = t, p, V
#     else:
#         R, G, B = V, p, q
#     R, G, B = R * 255, G * 255, B * 255
#     RGB = np.stack([R, G, B], axis=-1)
#     return RGB
#
# def RGB2HSV(RGB):
#
#     R, G, B = RGB[:,:,0], RGB[:,:,1], RGB[:,:,2]
#     mx = np.maximum(R,G,B)
#     mn = np.minimum(R,G,B)
#     df = mx - mn
#     if mx == mn:
#         H = 0
#     elif mx == R:
#         H = (60 * ((G - B) / df) + 360) % 360
#     elif mx == G:
#         H = (60 * ((B - R) / df) + 120) % 360
#     elif mx == B:
#         H = (60 * ((R - G) / df) + 240) % 360
#     if mx == 0:
#         S = 0
#     else:
#         S = df / mx
#     V = mx
#     HSV = np.stack([H, S, V], axis=-1)
#     return HSV
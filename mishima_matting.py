import scipy.spatial
import numpy as np

# Get fg/bg distances for each pixel from each surface on convex hull
def convex_hull_distance(cvx_hull, pixels):
    d_hull = np.ones(pixels.shape[0]*cvx_hull.equations.shape[0]).reshape(pixels.shape[0],cvx_hull.equations.shape[0])*1000
    for j, surface_eq in enumerate(cvx_hull.equations):
        for i, px_val in enumerate(pixels):
            nhat= surface_eq[:3]
            d_hull[i,j] = np.dot(nhat, px_val) + surface_eq[3]
    return  np.maximum(np.amax(d_hull, axis=1),0)

def mishima_matte(img, trimap):
    h,w,c = img.shape
    bg = trimap == 0
    fg = trimap == 255
    unknown = True ^ np.logical_or(fg,bg)
    fg_px = img[fg]
    bg_px = img[bg]
    unknown_px = img[unknown]

    # Setup convex hulls for fg & bg
    fg_hull = scipy.spatial.ConvexHull(fg_px)
    fg_vertices_px = fg_px[fg_hull.vertices]
    bg_hull = scipy.spatial.ConvexHull(bg_px)
    bg_vertices_px = bg_px[bg_hull.vertices]

    # Compute shortest distance for each pixel to the fg&bg convex hulls
    d_fg = convex_hull_distance(fg_hull, unknown_px)
    d_bg = convex_hull_distance(bg_hull, unknown_px)

    # Compute uknown region alphas and add to known fg.
    alphaPartial = d_bg/(d_bg+d_fg)
    alpha = unknown.astype(float).copy()
    alpha[alpha !=0] = alphaPartial
    alpha = alpha + fg
    return alpha

# Load in image
def main():    
    img  = scipy.misc.imread('toy.jpg')
    trimap = scipy.misc.imread('toyTrimap.png', flatten='True')

    alpha = mishima_matte(img, trimap)

    plt.imshow(alpha, cmap='gray')
    plt.show()
    h, w, c = img.shape
    plt.imshow((alpha.reshape(h,w,1).repeat(3,2)*img).astype(np.uint8))
    plt.show()

if __name__ == "__main__":
    import scipy.misc
    import matplotlib.pyplot as plt
    main()
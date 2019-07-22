def projectify(A,H):
    # cast input image to double
    A = np.asarray(A, dtype=np.float32)
    H = np.asarray(H, dtype=np.float32)
    # determine size of rows, cols and colour channels
    m, n, c = A.shape
    # determine size of output image by forward-transforming four corners of A
    p1 = hnorm(np.dot(H,np.asarray([1,1,1], dtype=np.float32)))
    p2 = hnorm(np.dot(H,np.asarray([n,1,1], dtype=np.float32)))
    p3 = hnorm(np.dot(H,np.asarray([1,m,1], dtype=np.float32)))
    p4 = hnorm(np.dot(H,np.asarray([n,m,1], dtype=np.float32)))
    minx = np.floor(np.min([p1[0],p2[0],p3[0],p4[0]]))
    maxx = np.ceil(np.max([p1[0],p2[0],p3[0],p4[0]]))
    miny = np.floor(np.min([p1[1],p2[1],p3[1],p4[1]]))
    maxy = np.ceil(np.max([p1[1],p2[1],p3[1],p4[1]]))
    nn = maxx - minx
    mm = maxy - miny
    print(nn,mm)
    # initialize output with white pixels
    B = np.zeros((int(mm),int(nn),int(c)))
    B.fill(250)
    # precompute the inverse of H
    Hi = np.linalg.inv(H)
    print(Hi)
    # loop through B's pixels
    print(nn,mm)
    for x in range(int(nn)):
        for y in range(int(mm)):
            # compensate for shift in B's origin
            p = np.asarray([x + minx - 1, y + miny - 1, 1])
            # apply inverse of H
            pp = hom_to_car(np.dot(Hi,p))
            xp = pp[0]
            yp = pp[1]
            # perform bilinear interpolation
            xpf = int(np.floor(xp))
            xpc = int(np.ceil(xp))
            if xpc >= n:
                xpc = n - 1
            elif xpf >= n:
                xpf = n - 1
            ypf = int(np.floor(yp))
            ypc = int(np.ceil(yp))
            if ypc >= m:
                ypc = m - 2
            elif ypf >= m:
                ypf = m - 2
            #print(ypf,xpf,ypc,xpf,ypf,xpc,ypc,xpc)
            # if (xpf > 0) and (xpc <= n) and (ypf > 0) and (ypc <= m):
            #     B[y,x,:] = np.dot(((xpc - xp)*(ypc - yp)),A[ypf,xpf,:]) + np.dot(((xpc - xp)*(yp - ypf)),A[ypc,xpf,:]) + np.dot(((xp - xpf)*(ypc - yp)),A[ypf,xpc,:]) + np.dot(((xp - xpf)*(yp - ypf)),A[ypc,xpc,:])

    B = np.asarray(B, dtype=np.int8)

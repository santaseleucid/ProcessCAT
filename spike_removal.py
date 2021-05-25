import numpy as np


def is_all_zero(dist):
    is_all_zero = np.all((dist == 0))
    return is_all_zero


def gen_dist(df, sample_dist):
    print("generating distance")
    length = df.shape[0]
    d = np.arange(0, sample_dist*length, sample_dist)
    df.dist = d
    return df


def spike_removal(rough_passed, dist_passed, curvature=-1e7, edge_slope=5e3, peak_factor=3.0):
    """
    Assumes dist_passed is in km 
    """
    rough = rough_passed
    dist = dist_passed * 1000  # km->m

    rows = np.size(dist)
    total_length = abs(dist[-1]-dist[0])
    samp_dist = abs(total_length/(rows-1.0))

    drdx = gen_drdx(rough, rows, samp_dist)
    d2rdx2 = gen_d2rdx2(rough, rows, samp_dist)

    start, stop = 1, rows-1

    # get sign of value to the left and right of every point n
    d1 = np.concatenate(([0], np.sign(drdx[start-1:stop-1]), [0]))
    d2 = np.concatenate(([0], np.sign(drdx[start+1:stop+1]), [0]))

    # look for locations where the sign on the left and right change
    inflections = np.logical_not(np.equal(d1,  d2))

    # look for locations where 2nd derivative is negative enough to warrant a spike
    y = d2rdx2 < curvature

    # list of points that fit the above criteria
    target_points = np.flatnonzero(np.logical_and(inflections, y))

    detected_spike = True  # secondary stopping condition
    num_spikes = 0  # sanity check variable - can be removed

    # while valid points and change has not stagnated
    while np.size(target_points > 0) and detected_spike:
        detected_spike = False
        for index in range(np.size(target_points)):
            n = target_points[index]
            # find first point satisfying condition
            p = np.argmax(abs(drdx[n-1::-1]) < edge_slope)
            # argmax returns 0 if can't find, checking if drdx[n-1-0] satisfies or argmax didn't find
            if p == 0 and drdx[n-1] >= edge_slope:
                p = n
            p = p + 1
            x1 = dist[n-p]

            # find first point satisfying condition
            q = np.argmax(abs(drdx[n+1:-1]) < edge_slope)
            # argmax returns 0 if can't find, checking if drdx[n+1+0] satisfies or argmax didn't find
            if q == 0 and drdx[n+1] >= edge_slope:
                q = rows-n
            q = q + 1
            x2 = dist[n+q]

            num = p+q-1
            change = (rough[n+q]-rough[n-p])/(num+1)
            height = (rough[n] - (rough[n-p] + change*p))*1e-6
            w = abs(x2-x1)
            if height > (w ** 2)/peak_factor:
                # spike detected
                detected_spike = True
                num_spikes = num_spikes + 1

                # interpolate
                interp = rough[n-p] + change*np.arange(1, num+1)
                rough[n-p+1:n-p+num+1] = interp

                # update sections in derivatives
                slice_low, slice_high = max(1, n-p), min(rows-1, n+q+1)
                drdx[slice_low:slice_high] = update_drdx(
                    rough, slice_low, slice_high, samp_dist)
                drdx[0] = drdx[1]
                drdx[-1] = drdx[-2]
                d2rdx2[slice_low:slice_high] = update_d2rdx2(
                    rough, slice_low, slice_high, samp_dist)

        # check list for new points of interest (if any)
        start, stop = 1, rows-1

        d1 = np.concatenate(([0], np.sign(drdx[start-1:stop-1]), [0]))
        d2 = np.concatenate(([0], np.sign(drdx[start+1:stop+1]), [0]))

        inflections = np.logical_not(np.equal(d1,  d2))
        y = d2rdx2 < curvature

        target_points = np.flatnonzero(np.logical_and(inflections, y))

    #stop_time = time.time()

    # print("PROCESSING DONE - TOTAL TIME: %f" % (stop_time - start_time))  # for supplied RMCatBatch run, was getting ~20 seconds to complete - can be removed
    return rough


def gen_drdx(rough, rows, samp_dist):
    # generate derivative of roughness w central difference method
    drdx = np.zeros_like(rough)

    start, stop = 1, rows-1

    drdx[start:stop] = (rough[start+1:stop+1] -
                        rough[start-1:stop-1])/(2.0*samp_dist)
    drdx[0] = drdx[1]
    drdx[-1] = drdx[-2]

    return drdx


def gen_d2rdx2(rough, rows, samp_dist):
    # generate 2nd derivative of roughness w central difference method
    d2rdx2 = np.zeros_like(rough)

    start, stop = 1, rows-2

    d2rdx2[start:stop] = (rough[start-1:stop-1] - 2 *
                          rough[start:stop] + rough[start+1:stop+1])/(samp_dist ** 2)

    return d2rdx2


def update_drdx(rough, slice_low, slice_high, samp_dist):
    # update subsection of roughness derivative when interpolating roughness section
    sub_drdx = (rough[slice_low+1:slice_high+1] -
                rough[slice_low-1:slice_high-1])/(2.0*samp_dist)
    return sub_drdx


def update_d2rdx2(rough, slice_low, slice_high, samp_dist):
    # update subsection of roughness 2nd derivative when interpolating roughness section
    sub_d2rdx2 = (rough[slice_low-1:slice_high-1] - 2*rough[slice_low:slice_high] +
                  rough[slice_low+1:slice_high+1])/(samp_dist ** 2)
    return sub_d2rdx2

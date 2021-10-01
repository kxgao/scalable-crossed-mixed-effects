import sys
import numpy as np


def colleastsq(VBhat, VEhat, p, colCounts, XTX, XTY, XTE, ETY):
    # computes column least squares estimate of beta

    XTEWETX = np.zeros((p, p))
    XTEWETY = np.zeros((p))
    for colid in colCounts.keys():
        XTEWETX = np.add(
            XTEWETX,
            np.outer(XTE[colid], XTE[colid] / float(VEhat + VBhat * colCounts[colid])),
        )
        XTEWETY = np.add(
            XTEWETY, XTE[colid] * ETY[colid] / float(VEhat + VBhat * colCounts[colid])
        )

    XTVBinvX = np.subtract(XTX / VEhat, VBhat * XTEWETX / VEhat)
    XTVBinvY = np.subtract(XTY / VEhat, VBhat * XTEWETY / VEhat)

    beta = np.linalg.solve(XTVBinvX, XTVBinvY)

    return beta, XTVBinvX


def rowleastsq(VAhat, VEhat, p, rowCounts, XTX, XTY, XTD, DTY):
    # computes row least squares estimate of beta

    XTDWDTX = np.zeros((p, p))
    XTDWDTY = np.zeros((p))
    for rowid in rowCounts.keys():
        XTDWDTX = np.add(
            XTDWDTX,
            np.outer(XTD[rowid], XTD[rowid] / float(VEhat + VAhat * rowCounts[rowid])),
        )
        XTDWDTY = np.add(
            XTDWDTY, XTD[rowid] * DTY[rowid] / float(VEhat + VAhat * rowCounts[rowid])
        )

    XTVAinvX = np.subtract(XTX / VEhat, VAhat * XTDWDTX / VEhat)
    XTVAinvY = np.subtract(XTY / VEhat, VAhat * XTDWDTY / VEhat)

    beta = np.linalg.solve(XTVAinvX, XTVAinvY)

    return beta, XTVAinvX


def moments(fileName):

    N = 0
    R = 0
    C = 0
    rowCounts = {}
    rowSums = {}
    rowDevs = {}
    colCounts = {}
    colSums = {}
    colDevs = {}
    total = 0
    totalDev = 0

    fileObj = open(fileName, "r")
    for line in fileObj:
        s = line.split()
        rowid = s[0]
        colid = s[1]
        val = float(s[2])

        if rowid not in rowCounts.keys():
            R += 1
            rowCounts[rowid] = 1
            rowSums[rowid] = val
            rowDevs[rowid] = 0
        else:
            rowCounts[rowid] += 1
            rowSums[rowid] += val
            rowDevs[rowid] += (rowCounts[rowid] * val - rowSums[rowid]) ** 2 / (
                rowCounts[rowid] * (rowCounts[rowid] - 1)
            )

        N += 1
        total += val
        if N > 1:
            totalDev += (N * val - total) ** 2 / (N * (N - 1))

        if colid not in colCounts.keys():
            colCounts[colid] = 1
            colSums[colid] = val
            colDevs[colid] = 0
            C += 1
        else:
            colCounts[colid] += 1
            colSums[colid] += val
            colDevs[colid] += (colCounts[colid] * val - colSums[colid]) ** 2 / (
                colCounts[colid] * (colCounts[colid] - 1)
            )
    fileObj.close()

    sumNid2 = sum([i ** 2 for i in rowCounts.values()])
    sumNdj2 = sum([j ** 2 for j in colCounts.values()])

    Ua = sum(rowDevs.values())
    Ub = sum(colDevs.values())
    Ue = N * totalDev

    M = np.zeros((3, 3))
    U = np.zeros((3))
    U[0] = Ua
    U[1] = Ub
    U[2] = Ue
    M[0, 1] = N - R
    M[0, 2] = N - R
    M[1, 0] = N - C
    M[1, 2] = N - C
    M[2, 0] = N ** 2 - sumNid2
    M[2, 1] = N ** 2 - sumNdj2
    M[2, 2] = N ** 2 - N

    theta = np.linalg.solve(M, U)
    VAhat = float(theta[0])
    VBhat = float(theta[1])
    VEhat = float(theta[2])

    if VAhat < 0:
        VAhat = 0.001
    if VBhat < 0:
        VBhat = 0.001
    if VEhat < 0:
        VEhat = 0.001

    ma = (3 * VBhat ** 2 + 12 * VBhat * VEhat + 3 * VEhat ** 2) * (N - R)
    mb = (3 * VAhat ** 2 + 12 * VAhat * VEhat + 3 * VEhat ** 2) * (N - C)
    me = (3 * VAhat ** 2 + 12 * VAhat * VEhat) * (N ** 2 - sumNid2)
    me += (3 * VBhat ** 2 + 12 * VBhat * VEhat) * (N ** 2 - sumNdj2)
    me += 3 * (N ** 2 - N) * VEhat ** 2 + 12 * VAhat * VBhat * (
        N ** 2 - sumNid2 - sumNdj2 + N
    )

    row4Devs = {}
    col4Devs = {}
    total4Dev = 0
    rowCross = 0
    colCross = 0

    fileObj = open(fileName, "r")
    for line in fileObj:
        s = line.split()
        rowid = s[0]
        colid = s[1]
        val = float(s[2])

        total4Dev += (val - float(total) / N) ** 4
        rowCross += float(colCounts[colid]) / rowCounts[rowid]
        colCross += float(rowCounts[rowid]) / colCounts[colid]

        if rowid not in row4Devs.keys():
            row4Devs[rowid] = (val - rowSums[rowid] / rowCounts[rowid]) ** 4
        else:
            row4Devs[rowid] += (val - rowSums[rowid] / rowCounts[rowid]) ** 4

        if colid not in col4Devs.keys():
            col4Devs[colid] = (val - colSums[colid] / colCounts[colid]) ** 4
        else:
            col4Devs[colid] += (val - colSums[colid] / colCounts[colid]) ** 4
    fileObj.close()

    Wa = sum(row4Devs.values()) + 3 * sum(
        [rowDevs[i] ** 2 / rowCounts[i] for i in rowCounts.keys()]
    )
    Wb = sum(col4Devs.values()) + 3 * sum(
        [colDevs[j] ** 2 / colCounts[j] for j in colCounts.keys()]
    )
    We = N * total4Dev + 3 * totalDev ** 2

    mu4rhs = np.zeros((3, 1))
    mu4rhs[0] = Wa - ma
    mu4rhs[1] = Wb - mb
    mu4rhs[2] = We - me
    mu4hat = np.linalg.solve(M, mu4rhs)

    muAhat = mu4hat[0]
    muBhat = mu4hat[1]
    muEhat = mu4hat[2]
    if muAhat < VAhat ** 2:
        muAhat = VAhat ** 2
    if muBhat < VBhat ** 2:
        muBhat = VBhat ** 2
    if muEhat < VEhat ** 2:
        muEhat = VEhat ** 2

    return VAhat, VBhat, VEhat, muAhat, muBhat, muEhat


def estimate(fileName):

    # Step 1 #

    N = 0
    R = 0
    C = 0
    rowCounts = {}
    colCounts = {}

    fileObj = open(fileName, "r")
    for line in fileObj:
        s = line.split()
        rowid = s[0]
        colid = s[1]
        try:
            val = float(s[2])
        except ValueError:
            print("Invalid response")
            break

        N += 1
        if N == 1:
            p = len(s) - 3
            XTX = np.zeros((p, p))
            XTY = np.zeros((p))
            XTD = {}
            DTY = {}
            XTE = {}
            ETY = {}
        pred = np.zeros((p))
        breaker = False
        for i in range(p):
            try:
                pred[i] = float(s[(3 + i)])
            except ValueError:
                print("Invalid predictor")
                breaker = True
                break
            if breaker:
                break
        prod = np.outer(pred, pred)
        XTX = np.add(XTX, prod)
        XTY = np.add(XTY, pred * val)

        if rowid in rowCounts.keys():
            rowCounts[rowid] += 1
            XTD[rowid] = np.add(XTD[rowid], pred)
            DTY[rowid] += val
        else:
            R += 1
            rowCounts[rowid] = 1
            XTD[rowid] = pred
            DTY[rowid] = val
        if colid in colCounts.keys():
            colCounts[colid] += 1
            XTE[colid] = np.add(XTE[colid], pred)
            ETY[colid] += val
        else:
            C += 1
            colCounts[colid] = 1
            XTE[colid] = pred
            ETY[colid] = val
    fileObj.close()

    beta = np.linalg.solve(XTX, XTY)
    maxrow = max(rowCounts.values())
    maxcol = max(colCounts.values())
    sumNid2 = sum([i ** 2 for i in rowCounts.values()])
    sumNdj2 = sum([j ** 2 for j in colCounts.values()])

    # Step 2 #

    fileObj = open(fileName, "r")
    txtout = open("diff.txt", "w")
    for line in fileObj:
        s = line.split()
        rowid = s[0]
        colid = s[1]
        try:
            val = float(s[2])
        except ValueError:
            print("Invalid response")
            break
        pred = np.zeros((p))
        breaker = False
        for j in range(p):
            try:
                pred[j] = float(s[3 + j])
            except ValueError:
                print("Invalid predictor")
                breaker = True
                break
            if breaker:
                break
        ymxb = val - np.dot(pred, beta)
        txtout.write("%s %s %s \n" % (rowid, colid, ymxb))
    fileObj.close()
    txtout.close()
    VAhat, VBhat, VEhat, muAhat, muBhat, muEhat = moments("diff.txt")

    # Step 3 #

    if VAhat * float(maxrow) >= VBhat * float(maxcol):
        rowcol = 1
        beta, XTVAinvX = rowleastsq(VAhat, VEhat, p, rowCounts, XTX, XTY, XTD, DTY)
    else:
        rowcol = 0
        beta, XTVBinvX = colleastsq(VBhat, VEhat, p, colCounts, XTX, XTY, XTE, ETY)

    # Step 4 #

    fileObj = open(fileName, "r")
    txtout = open("diff.txt", "w")
    for line in fileObj:
        s = line.split()
        rowid = s[0]
        colid = s[1]
        try:
            val = float(s[2])
        except ValueError:
            print("Invalid response")
            break
        pred = np.zeros((p))
        breaker = False
        for j in range(p):
            try:
                pred[j] = float(s[3 + j])
            except ValueError:
                print("Invalid predictor")
                breaker = True
                break
            if breaker:
                break
        ymxb = val - np.dot(pred, beta)
        txtout.write("%s %s %s \n" % (rowid, colid, ymxb))
    fileObj.close()
    txtout.close()
    VAhat, VBhat, VEhat, muAhat, muBhat, muEhat = moments("diff.txt")

    # asymptotic variance of variance components #
    M = np.zeros((3, 3))
    M[0, 0] = -1 / N
    M[0, 2] = 1 / (N ** 2)
    M[1, 1] = -1 / N
    M[1, 2] = 1 / (N ** 2)
    M[2, 0] = 1 / N
    M[2, 1] = 1 / N
    M[2, 2] = -1 / (N ** 2)
    M = np.matrix(M)

    varU = np.zeros((3, 3))
    varU[0, 0] = (muBhat - VBhat ** 2) * sumNdj2
    varU[1, 1] = (muAhat - VAhat ** 2) * sumNid2
    varU[2, 2] = (muAhat - VAhat ** 2) * sumNid2 * N ** 2 + (
        muBhat - VBhat ** 2
    ) * sumNdj2 * N ** 2
    varU[0, 1] = (muEhat - VEhat ** 2) * N
    varU[0, 2] = (muBhat - VBhat ** 2) * N * sumNdj2
    varU[1, 0] = varU[0, 1]
    varU[1, 2] = (muAhat - VAhat ** 2) * N * sumNid2
    varU[2, 0] = varU[0, 2]
    varU[2, 1] = varU[1, 2]
    varU = np.matrix(varU)
    varTheta = M * varU * M.T

    # asymptotic variance of beta estimate #

    if rowcol == 1:
        inv_XTVAinvX = np.linalg.inv(XTVAinvX)
        varXTVAinvB = np.zeros((p, p))
        colVals = {}
        fileObj = open(fileName, "r")
        for line in fileObj:
            s = line.split()
            rowid = s[0]
            colid = s[1]
            varXTVAinvB -= (
                2
                * VAhat
                * VBhat
                * np.outer(XTD[rowid], XTE[colid])
                / (VEhat ** 2 * (VEhat + VAhat * rowCounts[rowid]))
            )
            if colid in colVals.keys():
                colVals[colid] += XTD[rowid] / (VEhat + VAhat * rowCounts[rowid])
            else:
                colVals[colid] = XTD[rowid] / (VEhat + VAhat * rowCounts[rowid])
        fileObj.close()
        for colid in colVals.keys():
            varXTVAinvB += VBhat * np.outer(XTE[colid], XTE[colid]) / (VEhat ** 2)
            varXTVAinvB += (
                VBhat
                * VAhat ** 2
                * np.outer(colVals[colid], colVals[colid])
                / (VEhat ** 2)
            )
        varBeta = inv_XTVAinvX + np.dot(inv_XTVAinvX, np.dot(varXTVAinvB, inv_XTVAinvX))
    else:
        inv_XTVBinvX = np.linalg.inv(XTVBinvX)
        varXTVBinvA = np.zeros((p, p))
        rowVals = {}
        fileObj = open(fileName, "r")
        for line in fileObj:
            s = line.split()
            rowid = s[0]
            colid = s[1]
            varXTVBinvA -= (
                2
                * VBhat
                * VAhat
                * np.outer(XTE[colid], XTD[rowid])
                / (VEhat ** 2 * (VEhat + VBhat * colCounts[colid]))
            )
            if rowid in rowVals.keys():
                rowVals[rowid] += XTE[colid] / (VEhat + VBhat * colCounts[colid])
            else:
                rowVals[rowid] = XTE[colid] / (VEhat + VBhat * colCounts[colid])
        fileObj.close()
        for rowid in rowVals.keys():
            varXTVBinvA += VAhat * np.outer(XTD[rowid], XTD[rowid]) / (VEhat ** 2)
            varXTVBinvA += (
                VAhat
                * VBhat ** 2
                * np.outer(rowVals[rowid], rowVals[rowid])
                / (VEhat ** 2)
            )
        varBeta = inv_XTVBinvX + np.dot(inv_XTVBinvX, np.dot(varXTVBinvA, inv_XTVBinvX))

    print(
        "The estimates of the variance components are %f, %f, and %f."
        % (VAhat, VBhat, VEhat)
    )
    print(
        "The variances of those estimates are %f, %f, and %f."
        % (varTheta[0, 0], varTheta[1, 1], varTheta[2, 2])
    )
    print("The estimated regression coefficients are:")
    print(beta)
    print("Variances of the estimated regression coefficients are:")
    print(np.diagonal(varBeta))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        estimate(sys.argv[1])
    else:
        raise SystemExit("usage: python mixed.py <fileName.txt>")

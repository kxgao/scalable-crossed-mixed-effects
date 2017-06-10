import sys
import numpy as np

# STEP 1: first pass over data
# computes summary statistics about the data
# compute the entries of the matrix M
# compute U_a, U_b, and U_e
# compute the estimates of the variance components

# STEP 3: second pass over data
# compute the estimates of the kurtoses
# compute conservative variances of the estimates

def moments(fileName):

    # Step 1 #
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

    fileObj = open(fileName, 'r')
    for line in fileObj:
        s = line.split()
        rowid = s[0]
        colid = s[1]
        try:
            val = float(s[2])
        except ValueError:
            print('Invalid response')
            break
        
        if rowid not in rowCounts.keys():
            R +=1
            rowCounts[rowid] = 1
            rowSums[rowid] = val
            rowDevs[rowid]= 0
        else:
            rowCounts[rowid] += 1
            rowSums[rowid] += val
            rowDevs[rowid] += (rowCounts[rowid]*val-rowSums[rowid])**2/(rowCounts[rowid]*(rowCounts[rowid]-1))

        N += 1
        total += val
        if N > 1:
            totalDev += (N*val-total)**2/(N*(N-1))

        if colid not in colCounts.keys():
            colCounts[colid] = 1
            colSums[colid] = val
            colDevs[colid] = 0
            C += 1
        else:
            colCounts[colid] += 1
            colSums[colid] += val
            colDevs[colid] += (colCounts[colid]*val-colSums[colid])**2/(colCounts[colid]*(colCounts[colid]-1))
    fileObj.close()

    sumNid2 = sum([i**2 for i in rowCounts.values()])
    sumNdj2 = sum([j**2 for j in colCounts.values()])

    # STEP 2 #

    Ua = sum(rowDevs.values())
    Ub = sum(colDevs.values())
    Ue = N*totalDev
    
    M = np.zeros( (3,3))
    U = np.zeros( (3,1))

    U[0] = Ua
    U[1] = Ub
    U[2] = Ue
    M[0,1] = N-R
    M[0,2] = N-R
    M[1,0] = N-C
    M[1,2] = N-C
    M[2,0] = N**2 - sumNid2
    M[2,1] = N**2 - sumNdj2
    M[2,2] = N**2-N

    theta = np.linalg.solve(M,U)
    VAhat = theta[0]
    VBhat = theta[1]
    VEhat = theta[2]
    if VAhat < 0:
        VAhat = 0
    if VBhat < 0:
        VBhat = 0
    if VEhat < 0:
        VEhat = 0
    print("The estimates of the variance components are %f, %f, and %f." %(VAhat,VBhat,VEhat))

    # Step 3 #

    ma =  (3*VBhat**2+12*VBhat*VEhat+3*VEhat**2)*(N-R)
    mb =  (3*VAhat**2+12*VAhat*VEhat+3*VEhat**2)*(N-C)
    me =  (3*VAhat**2+12*VAhat*VEhat)*(N**2-sumNid2)
    me += (3*VBhat**2+12*VBhat*VEhat)*(N**2-sumNdj2)
    me += 3*(N**2-N)*VEhat**2+12*VAhat*VBhat*(N**2-sumNid2-sumNdj2+N)

    row4Devs = {}
    col4Devs = {}
    total4Dev = 0
    rowCross = 0
    colCross = 0
    rowTs = {}
    colTs = {}
    ZNm1m1 = 0
    ZNp1p1 = 0
    ZNm1p2 = 0
    ZNp2m1 = 0

    fileObj = open(fileName, 'r')
    for line in fileObj:
        s = line.split()
        rowid = s[0]
        colid = s[1]
        try:
            val = float(s[2])
        except ValueError:
            print('Invalid response')
            break

        total4Dev += (val-float(total)/N)**4
        rowCross += float(colCounts[colid])/rowCounts[rowid]
        colCross += float(rowCounts[rowid])/colCounts[colid]
        ZNm1m1 += 1.0/(rowCounts[rowid]*colCounts[colid])
        ZNp1p1 += rowCounts[rowid]*colCounts[colid]
        ZNm1p2 += colCounts[colid]**2/float(rowCounts[rowid])
        ZNp2m1 += rowCounts[rowid]**2/float(colCounts[colid])

        if rowid not in row4Devs.keys():
            row4Devs[rowid] = (val-rowSums[rowid]/rowCounts[rowid])**4
        else:
            row4Devs[rowid] += (val-rowSums[rowid]/rowCounts[rowid])**4
        if rowid not in rowTs.keys():
            rowTs[rowid] = colCounts[colid]
        else:
            rowTs[rowid] += colCounts[colid]

        if colid not in col4Devs.keys():
            col4Devs[colid] = (val-colSums[colid]/colCounts[colid])**4
        else:
            col4Devs[colid] += (val-colSums[colid]/colCounts[colid])**4
        if colid not in colTs.keys():
            colTs[colid] = rowCounts[rowid]
        else:
            colTs[colid] += rowCounts[rowid]
    fileObj.close()

    Wa = sum(row4Devs.values())+3*sum([rowDevs[i]**2/rowCounts[i] for i in rowCounts.keys()])
    Wb = sum(col4Devs.values())+3*sum([colDevs[j]**2/colCounts[j] for j in colCounts.keys()])
    We = N*total4Dev+3*totalDev**2

    mu4rhs = np.zeros((3,1))
    mu4rhs[0] = Wa-ma
    mu4rhs[1] = Wb-mb
    mu4rhs[2] = We-me
    mu4hat = np.linalg.solve(M,mu4rhs)

    muAhat = mu4hat[0]
    muBhat = mu4hat[1]
    muEhat = mu4hat[2]
    if muAhat<VAhat**2:
        muAhat = VAhat**2
    if muBhat<VBhat**2:
        muBhat = VBhat**2
    if muEhat<VEhat**2:
        muEhat = VEhat**2

    rowHarm = sum([1.0/i for i in rowCounts.values()])
    colHarm = sum([1.0/j for j in colCounts.values()])

    VarUa = (muBhat-VBhat**2)*(sumNdj2-rowCross)
    VarUa += 2*VBhat**2*rowCross
    VarUa += 4*VBhat*VEhat*(N-R)
    VarUa += (muEhat-VEhat**2)*(N+rowHarm-2*R)
    VarUa += 2*VEhat**2*(R-rowHarm)

    VarUb = (muAhat-VAhat**2)*(sumNid2-colCross)
    VarUb += 2*VAhat**2*colCross
    VarUb += 4*VAhat*VEhat*(N-C)
    VarUb += (muEhat-VEhat**2)*(N+colHarm-2*C)
    VarUb += 2*VEhat**2*(C-colHarm)

    VarUe = 2*VAhat**2*(sumNid2**2-sum([i**4 for i in rowCounts.values()]))
    VarUe += (muAhat-VAhat**2)*(N**2*sumNid2-2*N*sum([i**3 for i in rowCounts.values()])+sum([i**4 for i in rowCounts.values()]))
    VarUe += 2*VBhat**2*(sumNdj2**2-sum([j**4 for j in colCounts.values()]))
    VarUe += (muBhat-VBhat**2)*(N**2*sumNdj2-2*N*sum([j**3 for j in colCounts.values()])+sum([j**4 for j in colCounts.values()]))
    VarUe += 2*VEhat**2*N*(N-1)
    VarUe += (muEhat-VEhat**2)*N*(N-1)**2
    VarUe += 4*VAhat*VBhat*(N**3-2*N*ZNp1p1+sumNid2*sumNdj2)
    VarUe += 4*VAhat*VEhat*N*(N**2-sumNid2)
    VarUe += 4*VBhat*VEhat*N*(N**2-sumNdj2)

    CovUaUe = 2*VBhat**2*(sum([rowTs[i]**2/float(rowCounts[i]) for i in rowCounts.keys()])-ZNm1p2)
    CovUaUe += (muBhat-VBhat**2)*(N*sumNdj2-N*rowCross-sum([j**3 for j in colCounts.values()])+ZNm1p2)
    CovUaUe += 2*VEhat**2*(N-R)
    CovUaUe += (muEhat-VEhat**2)*(N-R)*(N-1)
    CovUaUe += 4*VBhat*VEhat*N*(N-R)

    CovUbUe = 2*VAhat**2*(sum([colTs[j]**2/float(colCounts[j]) for j in colCounts.keys()])-ZNp2m1)
    CovUbUe += (muAhat-VAhat**2)*(N*sumNid2-N*colCross-sum([i**3 for i in rowCounts.values()])+ZNp2m1)
    CovUbUe += 2*VEhat**2*(N-C)
    CovUbUe += (muEhat-VEhat**2)*(N-C)*(N-1)
    CovUbUe += 4*VAhat*VEhat*N*(N-C)

    CovUaUb = (muEhat-VEhat**2)*(N-R-C+ZNm1m1)

    varU = np.zeros( (3,3))
    varU[0,0] = VarUa
    varU[0,1] = CovUaUb
    varU[0,2] = CovUaUe
    varU[1,0] = CovUaUb
    varU[1,1] = VarUb
    varU[1,2] = CovUbUe
    varU[2,0] = CovUaUe
    varU[2,1] = CovUbUe
    varU[2,2] = VarUe

    Minv = np.mat(np.linalg.inv(M))
    varhattheta = Minv * varU * Minv.T
    VarVAhat = varhattheta[0,0]
    VarVBhat = varhattheta[1,1]
    VarVEhat = varhattheta[2,2]

    print("Conservative variances of the estimates are %f, %f, and %f." %(VarVAhat, VarVBhat, VarVEhat))

if __name__ == "__main__":
    if len(sys.argv)>1:
        moments(sys.argv[1])
    else:
        raise SystemExit("usage: python cre.py <fileName.txt>")

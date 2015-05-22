#!usr/bin/env python
# This script generates the nuclear attraction matrix elements over Gaussian basis functions on one center.
# As input, the script needs a partitioning of space in the radial direction. (ie. 2 3 for [2:3]).
# These boundaries designate the regions of space where the different quadratures are preformed.
# For the first region, [0:2], the integral is zero.  For the second region, [2:3], Gauss-Legendre quadrature is preformed
# for the nuclear attraction term times the partitioning function.
# For the third region of space, [3:\inf], the integral is preformed using Gauss-Laguerre quadrature (noting the
# weight factor of exp(-x))
#
# This script reads from the file expos.dat from the format of
#      0.0000  ss  2.30E3
# where the first term is the position of the center offset from the z-axis in Cartesian coordinates
# the next term is the symmetry and the last term is the exponent of the Gaussian basis function.
#
# The script writes to a file vnucpart.dat which can be used in the DVR insertion code.
import sys, math
import numpy


# Parameters denoting the nuclear charge attraction and the tolerance of the integration
z=-10.0
tol = 1.0E-10

# Read in the partitioning of space on the command line and check it's validity
try:
    rmin = numpy.float64(sys.argv[1])
    rmax = numpy.float64(sys.argv[2])
    if rmin < 0.0:
        print("rmin must not be negative!")
        sys.exit()
    if rmax < 0.0 or rmax < rmin:
        print("rmax must not be negative and must be bigger than rmin!")
        sys.exit()
except:
    print("Usage: ", sys.argv[0], " needs the the partitioning of space rmin, rmax")
    sys.exit()


# Define the partitioning function which smoothly goes from zero to one in the second region.
# Note that this function mathematically is piecewise in all of space: zero in the first region and one in the third.
def partition(x):
    global rmin
    global rmax

    y = numpy.float64((x-rmin)/(rmax-rmin))
    return numpy.float64(math.exp((y-1)*(y-1)/((y-1)*(y-1)-1)))

# Generate the Gauss-Legendre quadrature points and weights scaled to the second region.
def get_legendre(deg):
    global rmin
    global rmax

    # generate quadrature on grid [-1:1]
    x, w = numpy.polynomial.legendre.leggauss(deg)
    
    # transform them onto the grid [rmin, rmax]    
    xt = 0.5*(rmax-rmin)*(x + 1.0)+rmin
    wt = (rmax-rmin)*0.5*w

    return xt, wt

# Generate the Gauss-Laguerre quadrature points and weights scaled to the third region.
# Also returns the exponetional factor for shifting the domain of the integral
def get_laguerre(deg):
    global rmax

    # generate quadrature on grid [0:inf]
    x, w = numpy.polynomial.laguerre.laggauss(deg)
    
    # transform them onto the grid [rmin,inf]
    xt = x + rmax
    form = numpy.float64(math.exp(-rmax))

    return xt, w, form


# Check for the angular integrals over theta and phi.
# Returns the exponent of r and angular integral factor (without pi) from the Cartesian representation
# the angular integral returns none if the two functions are othogonal.
def angular(sym1, sym2):
    if sym1 == sym2:
        samesym = True
    else:
        samesym = False

    if samesym:
        if sym1 == 'ss':
            return 0, 4.0
        elif sym1 == 'px' or sym1 == 'py' or sym1 == 'pz':
            return 2, numpy.float64(4.0/3.0)
        elif sym1 == 'xx' or sym1 == 'yy' or sym1 == 'zz':
            return 4, 0.8
        elif sym1 == 'xy' or sym1 == 'xz' or sym1 == 'yz':
            return 4, numpy.float64(4.0/15.0)

    else:
        if sym1 == 'ss' and (sym2 == 'xx' or sym2 == 'yy' or sym2 == 'zz'):
            return 2, numpy.float64(4.0/3.0)
        elif sym2 == 'ss' and (sym1 == 'xx' or sym1 == 'yy' or sym1 == 'zz'):
            return 2, numpy.float64(4.0/3.0)
        else:
            return 0, None

# Preform the integral over the second region on un-normalized Guassian functions with the partitioning function
# Loop until it converges or needs 100 points
def gauss_legendre(rexpo, ab):
    # Quadrature r in [rmin,rmax] using Gauss-Legendre quadrature
    order = 50
    glsum1 = numpy.float64(0.0)
    glsum2 = numpy.float64(0.0)

    while(order < 100):

        x, w = get_legendre(order)
        for i in range(order):
            glsum1 += w[i]*math.exp(-ab*x[i]*x[i])*(x[i])**(1+rexpo)*partition(x[i])

        if( math.fabs(glsum1-glsum2) < tol):
            glsum2 = glsum1
            break
        else:
            glsum2 = glsum1
            glsum1 = numpy.float64(0.0)
            order += 5

#    print("The Legendre qaudrature converged at n = ", order, " at I = ", glsum2)
    return glsum2

# Preform the integral over the third region on un-normalized Guassian functions
# Loop until it converges or needs 100 points.
def gauss_laguerre(rexpo, ab):
    # Quadrature r in [rmax,inf] using Gauss-Laguerre quadrature
    order = 50
    lesum1 = numpy.float64(0.0)
    lesum2 = numpy.float64(0.0)
    while(order < 100):
    
        x, w, form = get_laguerre(order)
    
        for i in range(order):
            lesum1 += w[i]*math.exp(-ab*x[i]*x[i]+x[i])*(x[i])**(1+rexpo)

        if(math.fabs(lesum1-lesum2) < tol):
            lesum2 = lesum1
            break
        else:
            lesum2 = lesum1
            lesum1 = 0.0
            order += 5

    lesum2 *= form
#    print("The Laguerre qaudrature converged at n = ", order, " at I = ", lesum2)
    return lesum2


# Read in basis functions from expos.dat
basis = []
exposfile = open('expos.dat','r')

while True:
    line = exposfile.readline()
    if line is '':
        break
    splitline = line.split("\t")
    center = float(splitline[0])
    sym = splitline[1]
    expo = float(splitline[2])
    basis.append([center, sym, expo])

exposfile.close()

# Create the nuclear attraction matrix
vnuc = [[numpy.float64(0.0) for x in range(len(basis))]  for x in range(len(basis))]

# Loop over basis functions, laoding in the proper normalization factors that Mesa uses
for i in range(len(basis)):
    centera = basis[i][0]
    syma = basis[i][1]
    expoa = basis[i][2]

    if syma == 'ss':
        af = numpy.float64((2.0*expoa/math.pi)**(0.75))
    if syma == 'px' or syma == 'py' or syma == 'pz':
        af = numpy.float64((2.0**7*expoa**5/math.pi**3)**(0.25))
    if syma == 'xx' or syma == 'yy' or syma =='zz' or syma == 'xy' or syma =='xz' or syma == 'yz':
        af = numpy.float64((2.0**11*expoa**7/math.pi**3)**(0.25))

    # Second loop over basis functions
    for j in range(i,len(basis)):
        centerb = basis[j][0]
        symb = basis[j][1]
        expob = basis[j][2]
        
        if symb == 'ss':
            bf = numpy.float64((2.0*expob/math.pi)**(0.75))
        if symb == 'px' or symb == 'py' or symb == 'pz':
            bf = numpy.float64((2.0**7*expob**5/math.pi**3)**(0.25))
        if symb == 'xx' or symb == 'yy' or symb =='zz' or symb == 'xy' or symb =='xz' or symb == 'yz':
            bf = numpy.float64((2.0**11*expob**7/math.pi**3)**(0.25))

        # Test orthogonality and read in the angular factors
        rexpo, angfac = angular(syma, symb)

        # If Orthogonal
        if angfac is None:
            vnuc[i][j] = "{:.9E}".format(numpy.float64(0.0))

        # Call integration routines if non-orthogonal
        else:
            ab = numpy.float64(expoa+expob)
            integral = gauss_legendre(rexpo, ab)
            integral += gauss_laguerre(rexpo, ab)
            integral *= z*angfac*math.pi*af*bf
            # If the integral is very small, make zero
            if (math.fabs(integral) < tol):
                integral = 0.0
            vnuc[i][j] = "{:.9E}".format(numpy.float64(integral))
            #print("This is the integral for ", expoa, syma, "and ", expob, symb, ": ", integral)

        # Exploit the fact that the nuclear attraction matrix is symmetric
        if i is not j:
            vnuc[j][i] = vnuc[i][j]


# Read in the xform.dat file for the contractions.
contract = [ ]
xformfile = open('xform.dat', 'r')
contdat = xformfile.readlines()

for i in range(len(contdat)):
    line = contdat[i].split(" ")
    contract.append([float(j) for j in line if j is not ''])

xformfile.close()

xformmat = numpy.matrix(contract)
vmat = numpy.matrix(vnuc, float)

temp = numpy.dot(vmat,numpy.transpose(xformmat))
vxformed = numpy.dot(xformmat,temp)

temp = vxformed.tolist()

# Write matrix out to file to be read by the radial DVR insertion code.
dirfile = open('vnucpart.dat', 'w')
for i in range(len(vxformed)):
    line = '  '.join(str('%.8E'%j) for j in temp[i])
    dirfile.write(line+'\n')
dirfile.close()

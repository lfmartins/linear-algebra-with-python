import numpy as np
import sympy
import re
from enum import Enum

def rational_matrix(matrix_a):
    array_a = np.array(matrix_a)
    ls = len(array_a.shape)
    if ls == 2:
        return sympy.Matrix([[sympy.nsimplify(elem) for elem in row] for row in array_a])
    if ls == 1:
        return sympy.Matrix([sympy.nsimplify(elem) for elem in array_a])
    raise ValueError('input must be an array of dimension at most two')

# Regular expression for row operations. Matches the following three kinds of expression:
#
# R<int> <=> R<int>
# R<int> * (<factor>) => R<int>
# R<int> * (<factor>) + R<int> => R<int>
#
# where:
# <int> is a decimal integer
# <factor> is either a <decimal> or <decimal>/<decimal>, where <decimal> is a decimal number.
pattern = r'''^
 \s* R(?P<source_row> \d+)                          # Source row
 \s* (
      ( <=> \s* R(?P<swap_row> \d+))                # Swap row
      | 
      ( 
        \*                                          # Literal * 
        \s* \(                                      # Literal (
        \s* (?P<num>[+-]?((\d+(\.\d*)?)|(\.\d+)))   # Numerator
        (                                           # Optional denominator
          \s* /                                     # Literal /
          \s* (?P<den>[+-]?((\d+(\.\d*)?)|(\.\d+))) # Denominator
        )?                                          
        \s* \)                                      # Literal )
        (                                           # Optional add row
          \s* \+                                    # Literal +
          \s* R(?P<add_row> \d+)                    # Add row
        )?
        \s* => \s* R(?P<target_row> \d+)            # Target row
      )
     )
\s* $
'''

re_rop = re.compile(pattern, re.I | re.X)

class Op(Enum):
    swap = 0
    scale = 1
    scale_add = 2

def rop_compile(rop_str):
    def error_msg(error_string):
        return (error_string + ': {}').format(rop_str)
    m = re_rop.match(rop_str)
    if m is None:
        raise ValueError(error_msg('invalid row operation'))
    md = m.groupdict()
    nsource = int(md['source_row']) - 1
    if md['swap_row'] is not None:
        nswap = int(md['swap_row']) - 1
        return (Op.swap, nsource, nswap, 0)
    nnum = eval(md['num'])
    nden = 1 if md['den'] is None else eval(md['den'])
    if nden == 0:
        raise ValueError(error_msg('zero denominator in row operation'))
    c = sympy.Rational(nnum, nden)
    ntarget = int(md['target_row']) - 1
    if md['add_row'] is None:
        if nsource != ntarget:
            raise ValueError(error_msg('source and target rows are not the same'))
        if c == 0:
            raise ValueError(error_msg('zero scaling factor in row operation'))
        return (Op.scale, nsource, ntarget, c)
    nadd = int(md['add_row']) - 1
    if nadd != ntarget:
        raise ValueError(error_msg('add and target rows are not the same'))
    if nadd == nsource:
        raise ValueError(error_msg('add and source rows are the same'))
    return (Op.scale_add, nsource, nadd, c)

def rop_swap(matrix_a, i, j, inplace=False):
    matrix_b = matrix_a if inplace else matrix_a[:, :]
    matrix_b[i, :], matrix_b[j, :] = matrix_b[j, :], matrix_b[i, :]
    return matrix_b


def rop_scale(matrix_a, i, c, inplace=False):
    matrix_b = matrix_a if inplace else matrix_a[:, :]
    matrix_b[i, :] = c * matrix_b[i, :]
    return matrix_b


def rop_scale_add(matrix_a, i, c, j, inplace=False):
    matrix_b = matrix_a if inplace else matrix_a[:, :]
    matrix_b[j, :] += c * matrix_b[i, :]
    return matrix_b


def do_rop(matrix_a, t, r1, r2, c, inplace=False):
    case_dict = {
        Op.swap: lambda a, r1, r2, c, inplace: rop_swap(a, r1, r2, inplace),
        Op.scale: lambda a, r1, r2, c, inplace: rop_scale(a, r1, c, inplace),
        Op.scale_add: lambda a, r1, r2, c, inplace: rop_scale_add(a, r1, c, r2, inplace)
    }
    return case_dict[t](matrix_a, r1, r2, c, inplace)


def rop(matrix_a, *rop_seq, inplace=False):
    matrix_b = matrix_a if inplace else matrix_a[:, :]
    for rop_str in rop_seq:
        t, r1, r2, c = rop_compile(rop_str)
        do_rop(matrix_b, t, r1, r2, c, inplace=True)
    return matrix_b


def reduced_row_echelon_form(matrix_a, inplace=False, ropseq=False, extra_cols=0):
    matrix_b = matrix_a if inplace else matrix_a[:, :]
    m, n = matrix_b.shape
    j = -1
    latex_str = ''
    if ropseq:
        rseq = []
    for i in range(m):
        j += 1
        while j < n - extra_cols:
            for k in range(i, m):
                if matrix_b[k, j] != 0:
                    break
            else:
                j += 1
                continue
            break
        if j >= n - extra_cols:
            break
        if k != i:
            rop_swap(matrix_b, i, k, inplace=True)
            if ropseq:
                rseq.append('R{}<=>R{}'.format(i+1, k+1))
        c = matrix_b[i,j] ** (-1)
        rop_scale(matrix_b, i, c, inplace=True)
        if ropseq:
            rseq.append('R{}*({})=>R{}'.format(i+1, c, i+1))
        for k in range(0, m):
            if k == i:
                continue
            c = -matrix_b[k,j]
            rop_scale_add(matrix_b, i, c, k, inplace=True)
            if ropseq:
                rseq.append('R{}*({})+R{}=>R{}'.format(i+1, c, k+1, k+1))
    if ropseq:
        return matrix_b, rseq
    return matrix_b


def matrix_to_system_latex(A, vnames=None):
    m, n = A.shape
    if vnames is None:
        vnames = ','.join('x{}'.format(i+1) for i in range(n))
    #print(vnames)
    vs = sympy.symbols(vnames)
    sout = '\\begin{{alignat*}}{{{}}}\n'.format(2*n+1)
    for i in range(m):
        leading = True
        for j in range(n):
            a = A[i,j]
            if j < n - 1:
                v = vs[j]
                if a == 0:
                    sout += '&{}{}&'
                    continue
                if leading:
                    if a != 0:
                        sout += sympy.latex(a*v)
                        leading = False
                    else:
                        sout += '&{}{}&'
                else:
                    pm = '-' if a < 0 else '+' if not leading else ''
                    sout += ' &{}' + pm + '{}& ' + sympy.latex(abs(a)*v)
            else:
                if leading:
                    sout += '0'
                sout += ' &{{}}={{}}&{}'.format(sympy.latex(a))
        sout += '\\\\ \n'
        
    sout += '\end{alignat*}\n'
    return sout

if __name__=='__main__':
    print('''This file is not meant to be executed. 
Load it to your script or Jupyter notebook using:

from latools import *

or:

import latools

The file must be in the same directory as the script or notebook loading it,
or in a path-accessible directory.''')





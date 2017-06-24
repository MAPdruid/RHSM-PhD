# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 14:59:59 2010

@author: joseph.wright
"""

from __future__ import division

import numpy 
import os 
import shutil
import copy 
import time
import sys
import math
from math import cos, sin, radians, atan, atan2, tan, degrees, pi
from re import sub
 
#3rd party packages
import tables
import arcpy

#My packages 
import condat



class HSM(object):
    """Super class of hierarchical surface models
    This version uses numpy arrays, stored in a PyTable tree to store the data"""

    def __init__(self, fname):
        #create the HHSM instance
        self.fname = fname
        self.cellsize = 1
        self.rotation = 0
        self.NoDataValue = -9999
        self.level = 0
        self.agg = 0
        self.x0 = 0
        self.y0 = 0
        #projection store as unicode object because Spatial References cannot be deep copied
        self.projection = u'unknown'
        self.h5 = '{0}.h5'.format(self.fname)

#‘r’: Read-only; no data can be modified.
#‘w’: Write; a new file is created (an existing file with the same name would be deleted).
#‘a’: Append; an existing file is opened for reading and writing, and if the file does not exist it is created.
#‘r+’: It is similar to ‘a’, but the file must already exist.        
        #open/create the h5file. If it exist it will be opened, otherwise created
        if os.path.exists(self.h5):           
            arcpy.AddWarning('{0} exists'.format(self.h5))
            with tables.openFile(self.h5, mode = "r") as h5file:
                #print h5file
                root = h5file.getNode("/L0")
                self.fname = root._v_attrs.fname
                self.cellsize = root._v_attrs.cellsize
                self.rotation = root._v_attrs.rotation
                self.NoDataValue = root._v_attrs.NoDataValue
                self.level = root._v_attrs.level 
                self.agg = root._v_attrs.agg
                self.x0 = root._v_attrs.x0
                self.y0 = root._v_attrs.y0
                self.projection = root._v_attrs.projection
                self.h5 = root._v_attrs.h5
                
        else:
            with tables.openFile(self.h5, mode = "w", title = os.path.basename(self.fname)) as h5file:
                pass
            

            
    def _HIP2HHSM(self, hip, level=None, agg=None):
    #Input: Tuple HIP index
    #Output: (where (str), name (str), index (tup))
    #ie ('/1/2', '3', (4,5,6)
    #Use this tool to process HIP indexes into the path and index of a single cell in an HHSM array tree. The output does not include the level indicator.
        if not level:
            level=self.level
        if not agg:
            agg=min(self.agg, self.level)
            
        tree=level - agg
        
        if tree:
            where=''.join('/'+str(x) for x in hip[:tree-1])
            #print hip, tree
            name=str(hip[tree-1])
        else:
            #The whole HIP is in the array position therefore a dummy '0' is used as name
            name='0'
            where = ''
        index = hip[tree:]
        
        return (where, name, index)
        
    def _HHSM2HIP(self, where, name, index):
        #reverse of the above
        if self.level - self.agg:
            return tuple(int(x) for x in where.split('/')[1:]) + (int(name),) + index
        else:
            return index 
            
    def _set_edge_nodata(self, level=0):
        #Changes the values of any edge cells to no data value
        calc = calcHSM(self)
        with tables.openFile(self.h5, mode = "r+") as h5file: 
            for array in h5file.walkNodes("/L{0}".format(level), "Array"):
                np = array.read()
                it =  numpy.nditer(np, flags=['multi_index'], op_flags=['readwrite'])
                while not it.finished:
                    IP = tuple (int(val) for val in array.title[:-self.agg])+it.multi_index
                    for i in range(self.aperture):
                        hood_IP = calc.IPadd(IP, (i,), self.level-level)
                        if len(hood_IP) > len(IP):
                            it[0] = self.NoDataValue
                    it.iternext()
                array[:] = np
                array.flush()
    
    def _convIP2xy(self, ip):      
        result = numpy.array([[0],[0]])
        for i, val in enumerate(ip[::-1]):
            result = self._IP2xy(result, i, val)
        return result
        
    def _NoDataCheck(self, val):
        return abs(self.NoDataValue-val) < abs(self.NoDataValue*0.000001)
            
    def CreateHIP(self):
        #Creates an HIP datastructure (h_val) of given level, aggregate
        #levels above the agg are nodes, below are dimensions of a numpy array
        #values are set to self.NoData value. 
        #The origin is the centre of the centre hexagon
        #automatically deletes and recreates if a file with the same name already exist
        
        arcpy.AddMessage('Creating HSM: {0}'.format(self.h5))
        #force agg to be <= level
        self.agg = min(self.level, self.agg)
        #Make file        
        with tables.openFile(self.h5, mode = "w", title = os.path.basename(self.fname)) as h5file:        
            #Make groups/tables            
            if self.level - self.agg > 0:
                tree_array_shape = (self.aperture,)*(self.level-self.agg)
                tree_array = numpy.zeros(tree_array_shape)
            else:
                tree_array = numpy.zeros((1,))
            it = numpy.nditer(tree_array, flags=['multi_index'])            
            #create the agg numpy array            
            np_array = numpy.zeros((self.aperture,)*min(self.level, self.agg),)
            #initialise to NoData value
            np_array[:] = self.NoDataValue
            arcpy.AddMessage('level: {0}, Agg: {1}'.format(self.level, self.agg))
            arcpy.AddMessage('Tree size: {0}'.format(tree_array.shape))
            arcpy.AddMessage('Array size: {0}'.format(np_array.shape))
            arcpy.AddMessage('Array type: {0}\n'.format(np_array.dtype))
            del tree_array
            
            #for each node insert a copy of the array
            while not it.finished:
                #make filename variables
                array_name = str(it.multi_index[-1:])[1:-2]
                #table name is a text digit of HIP indice eg '0'
                array_where ="/L0"+''.join('/'+str(x) for x in (it.multi_index)[:-1])
                #where is the node location as file name eg'/L0/0
                array_title = (''.join(str(x) for x in (it.multi_index))+'0'*self.agg)[-self.level:]
                #the title is the translation ie '0100' of the array to its location in the hhsm
                #The special case of the agg = level requires the removal of the additional zero
                #Insert the array into the group
                table = h5file.createArray(array_where, array_name, np_array, title =  array_title, createparents=True)
                table.flush()
                it.iternext()
            del it
            #add the header attributes. Add to /L0
            root = h5file.getNode("/L0")
            root._v_attrs.fname = self.fname
            root._v_attrs.cellsize = self.cellsize
            root._v_attrs.rotation = self.rotation
            root._v_attrs.NoDataValue = self.NoDataValue
            root._v_attrs.level = self.level
            root._v_attrs.agg = self.agg
            root._v_attrs.x0 = self.x0
            root._v_attrs.y0 = self.y0
            root._v_attrs.projection = self.projection
            root._v_attrs.h5 = self.h5 

    def ValuesFromDEM(self, inputDEM, sampling):
        #changes the h_values of the HHSM instance to values from the inputDEM
        #If sampling = centre, at the centre point of each hexagon
        ##If sampling = corners, the average of the corners and centre, change weighting 3x for centre
        #If sampling =  Areal Projection, first order least squares spline match (aereal projection)
        #inputDEM is text filename (eg r'filepath/filename')
        #Saves values to the H5 file
        #H5 file must already exist
        
        arcpy.AddMessage("Extracting values from {0}".format(inputDEM))
        NPR = NumPyRaster(inputDEM) 
        NPR.StripNoDataEdges()
        self.NoDataValue = NPR.no_data_value
                
        if sampling == "Areal Projection":
            #need to fix
           self.__FirstOrder__(inputDEM)
        
        elif sampling.lower() == "corners":
            #need to fix
            calc = calcHSM(self)
            with tables.openFile(self.h5, mode = "r+") as h5file:
                for array in h5file.walkNodes("/L0", "Array"):
                    np = array.read()
                    it =  numpy.nditer(np, flags=['multi_index'], op_flags=['readwrite'])
                    while not it.finished:
                        corners = calc.xy_corners(tuple (int(val) for val in array.title[:-self.agg])+it.multi_index)
                        extent_check = 1
                        total = 0
                        for corner in corners:
                            x, y = corner
                            if not NPR.x0 < x < NPR.x0 + NPR.cellsizex*NPR.data.shape[0] or not NPR.y0 < y < NPR.y0 + NPR.cellsizey*NPR.data.shape[1]: #if the cell is outside the extent of the input data fail the extent
                                extent_check = 0
                            else:
                                xcell, ycell  = int((x-NPR.x0)/NPR.cellsizex), int((y-NPR.y0)/NPR.cellsizey)
                                total += NPR.data[xcell, ycell]
                        if not extent_check:
                            it[0]= self.NoDataValue
                        else:
                            x, y = corners[0]
                            xcell, ycell  = int((x-NPR.x0)/NPR.cellsizex), int((y-NPR.y0)/NPR.cellsizey)
                            total += 2*NPR.data[xcell, ycell]#give the centre 3* weight
                            it[0] = float(total)/(len(corners)+2)
                            #arcpy.AddMessage(float(total)/(len(corners)+2))
                        it.iternext()
                        
                    del it
                    array[:]=np
                    array.flush()
                
        elif sampling.lower() == "centre":
            #applies the value of the DEM ata the centre of the hexagon
            #print self.NoDataValue
            with tables.openFile(self.h5, mode = "r+") as h5file:
                #create a conversion object
                conv = calcHSM(self)
                for array in h5file.walkNodes("/L0", "Array"):
                    np = array.read()
                    it =  numpy.nditer(np, flags=['multi_index'], op_flags=['readwrite'])
                    while not it.finished:
                        HIP = tuple (int(val) for val in array.title[:-self.agg])+it.multi_index
                        (x, y) = conv.IP2xy(HIP)
                        if not NPR.x0 < x < NPR.x0 + NPR.cellsizex*NPR.data.shape[0] or not NPR.y0 < y < NPR.y0 + NPR.cellsizey*NPR.data.shape[1]:
                            it[0] = self.NoDataValue
                        else:
                            xcell, ycell  = int((x-NPR.x0)/NPR.cellsizex), int((y-NPR.y0)/NPR.cellsizey)
                            it[0]  = NPR.data[xcell, ycell] #could update array here 
                        it.iternext()
                    
                    array[:]=np
                    array.flush()

                    
    def ValuesFromTIN(self, inputTIN, sampling = "LINEAR"):
        #Extracts values from a TIN using the sampling provided from ESRI
        arcpy.AddMessage("Extracting values from {0}".format(inputTIN))
        arcpy.CheckOutExtension("3D")
        #for each array create a point layer
        temp_file = self.fname +'_TINtemp'
        gdb, fname = os.path.split(temp_file)
        calc = calcHSM(self)
        arcpy.CreateFeatureclass_management(gdb, fname, 'POINT')
        with tables.openFile(self.h5, mode = "r+") as h5file:
            for array in h5file.walkNodes("/L0", "Array"):
                np = array.read()
                it =  numpy.nditer(np, flags=['multi_index'], op_flags=['readwrite'])                
                arcpy.AddField_management (temp_file, 'MULTI', 'TEXT')

                #create points with the geometry and the array addy
                cur = arcpy.da.InsertCursor (temp_file, ("MULTI", "SHAPE@XY"))
                while not it.finished:
                    #create a string version of multi index
                    IP_text = ''.join(str(x) for x in (it.multi_index))
                    #Calc (x,y) position
                    xy = calc.IP2xy(tuple (int(val) for val in array.title[:-self.agg])+it.multi_index)
#                    print IP_text, xy
                    cur.insertRow([IP_text, xy])
                    it.iternext() 
                del it
                del cur
                #add shape properties from tin
                arcpy.AddSurfaceInformation_3d (temp_file, inputTIN, "Z", sampling)
                #iterate the points storing the value in the array
                rows = arcpy.da.SearchCursor (temp_file, ["MULTI", "Z"])
                for row in rows:
                    IP = tuple (int(val) for val in row[0])
                    if row[1] is None:
                        np[IP] = self.NoDataValue
                    else:
                        np[IP] = row[1] 
                del rows
                #update data
                array[:]=np
                array.flush()
                           
                #delete the points
                arcpy.DeleteRows_management(temp_file)
        #delete the temp fc
        arcpy.Delete_management(temp_file)
                    
    def valuesFromEq(self, equation, limit = None):
        #determines the values from the input equation.
        #form eq as a python z(x,y) expression of x and y as text ie 'x**2 +y**2'
        #limit is an an equation representing the area which will have a data value
        
        arcpy.AddMessage('Applying equation y = {0}'.format(equation))
        calc = calcHSM(self)
        #Iterate through the array
        with tables.openFile(self.h5, mode = "r+") as h5file:
            for array in h5file.walkNodes("/L0", "Array"):
                np = array.read()
                it = numpy.nditer(np, flags=['multi_index'], op_flags=['readwrite'])
                while not it.finished:
                    #for each cell determine the xy coords
                    x, y  = calc.IP2xy(self._HHSM2HIP(array._v_parent._v_pathname[2:], array.name, it.multi_index))#note drop the /L0 from the pathname
                    #apply the eq (must be of x, y)
                    if limit:
                        if eval(limit):
                           value = eval(equation)
                        else:
                            value = self.NoDataValue
                    else:
                        value = eval(equation)
                    if value == numpy.inf:
                        value = self.NoDataValue

                    #set the value
                    it[0] = value
                    it.iternext()
                del it
                #update data
                array[:]=np
                array.flush()
                

                                              

    def Copy(self, filename, values = False):
        """Creates a copy of the dataset with the given filename
        If values = False the datafile is not copied
        otherwise the datafile is also copied"""
        
        #create the output HSM
        new_copy = copy.deepcopy(self)
        new_copy.h5 = '{0}.h5'.format(filename)
        new_copy.fname = filename
        
        if values == False:
            #delete h5 file if one exists with the name of the copy's .h5
            if os.path.exists(new_copy.h5):
                os.remove(new_copy.h5)
            #true_HSM.CreateHIP()
        elif values == True:
            shutil.copyfile(self.h5, '{0}.h5'.format(filename))
        
        return new_copy
                    
    def PyrTree(self, operation = 'mean', stop = None, loop = None, error = False):
        """Adds branches to the h5 file for each pyramid level up to and including stop level (L1 to Ln). Each branch has the same stucture and agg value the base level (L0), however each level is progressively smaller. Each pyrimid level is a summary of all the values in the level immediately below, all values are initiated as 0 (should be changed to NoData).
        Automatically deletes and recreates if a branch with the same name already exist
        Operation is the method of summary
        Stop allows the ability to stop processing at a finer level. There is no coresponding start because the lower levels generally need to be calculated anyway.
        If loop is true, include as a tuple the values which are equivalent to 0 (ie (1, 7) for RHSM-hex (0, 2*pi) for radians)
        If error is true the maximum discrepency between the summary value and the fine res values is stored in a seperate tree for each level.
        It is necesary that L0 already contains values."""

        if not stop:
            stop = self.level
            


        #Open the file            
        with tables.openFile(self.h5, mode = "r+", title = os.path.basename(self.fname)) as h5file:
            #delete existing
            for OldL in range(1, self.level+1):
                if '/L{0}'.format(OldL) in h5file:
                    h5file.removeNode('/', '/L{0}'.format(OldL), recursive = True)
                    print "Warning: Existing /L{0} was deleted".format(OldL)
                if '/E{0}'.format(OldL) in h5file:
                    h5file.removeNode('/', '/E{0}'.format(OldL), recursive = True)
                    print "Warning: Existing /E{0} was deleted".format(OldL)
            
            for L in range(1,stop+1):#for each level of the pyramid to be drawn
            
                #determine the size of the .h5 tree and create an iterator.
                arcpy.AddMessage('Calculating {0} Pyramid Level {1}'.format(operation, L))
                if self.level-L - self.agg > 0:
                    group_array = numpy.zeros((self.aperture,)*(self.level-self.agg-L))
                else:
                    group_array = numpy.zeros((1,))
                it = numpy.nditer(group_array, flags=['multi_index'])
                arcpy.AddMessage('Tree size: {0}'.format(group_array.shape))
                del group_array
                                
                #create a blank array
                coarse_array = numpy.array(numpy.zeros((self.aperture,)*min(self.agg, self.level-L)))                
                if coarse_array.shape == ():#Avoid 0-shape arrays
                    coarse_array = coarse_array.reshape(1,)
                if error:
                    error_array = numpy.array(numpy.zeros((self.aperture,)*min(self.agg, self.level-L)))                
                    if error_array.shape == ():#Avoid 0-shape arrays
                        error_array = coarse_array.reshape(1,)
                    #need the base array for comparison
                    #This will be hard may need different arrays
                    #Can the previous error numbers be used?
                    #between the shifting mean and the loop it is not possible to record high and low values
                    base_array = h5file.getNode('/L0/0').read()#hard coded work around only for agg = level HSMs
                    
                arcpy.AddMessage('Array size: {0}\n'.format(coarse_array.shape))
                
                #for each leaf in the tree create an array that is the operation output of the corresponding area of the level below
                while not it.finished:
                    #define the file name varaiables                    
                    coarse_where ="/L{0}".format(L) + ''.join('/'+str(x) for x in (it.multi_index)[:-1])
                    if self.level - self.agg - L > 0:
                        fine_where ="/L{0}".format(L-1) + ''.join('/'+str(x) for x in (it.multi_index))
                    else:
                        fine_where ="/L{0}".format(L-1) + ''.join('/'+str(x) for x in (it.multi_index)[:-1])
                    coarse_name = str(it.multi_index[-1:])[1:-2]
                    if error:
                        error_where = "/E{0}".format(L) + ''.join('/'+str(x) for x in (it.multi_index)[:-1])
                        error_name = str(it.multi_index[-1:])[1:-2]
                    
                
                                         
                    #Determine the operation results and place them in the coarse array
                    for array in h5file.walkNodes(fine_where, "Array"):
                        np_array = array.read()
                        #mask out Nodata cells and calculate means
                        axis = len(np_array.shape) - 1                            
                        nodata = numpy.equal(np_array, self.NoDataValue)
                        coarse_nodata = nodata.any(axis)
                        
                        if loop:#closed arithmetc
                            #determine difference from upper and lower limit
                            lower = np_array - loop[0]
                            upper = np_array - loop[1]
                            #substitute the lower limit difference with the upper limit difference if the upper limit is closer
                            swapped = lower.copy()
                            swapped[abs(upper) < abs(lower)] = upper[abs(upper) < abs(lower)]
                            #has the substitution lowered variace?
                            var_bool = swapped.var(axis) < lower.var(axis)
                            lower[var_bool] = swapped[var_bool]
                            #calculate the mean 
                            coarse_data =  numpy.ma.array(lower, mask = nodata).mean(axis)
                            #put means back into the interval
                            coarse_data = ((coarse_data)%(loop[1] - loop[0])) + loop[0]
                            #calculate the mean and put it back into the interval
                            #coarse_data = (numpy.ma.array(lower, mask = nodata).mean(axis) + loop[0])%(loop[1] - loop[0]) + loop[0]
                        else:
                            coarse_data = numpy.ma.array(np_array, mask= nodata).mean(axis)
                        
                        if len (h5file.listNodes(fine_where, "Array"))==1:
                            coarse_array = numpy.ma.array(coarse_data, mask = coarse_nodata).filled(self.NoDataValue)
                            if coarse_array.shape == ():#Avoid 0-shape arrays
                                coarse_array = coarse_array.reshape(1,)
                            if error:
                                #use the base layer values to determine error not np_array
                                if loop:
                                    inter = loop[1]-loop[0] 
                                    diff = (base_array - coarse_data.reshape(coarse_data.shape + (1L,)*L) + (inter)/2.0)%inter - inter/2.0
                                else:
                                    diff = base_array - coarse_data.reshape(coarse_data.shape + (1L,)*L) 
                                for diff_axis in range(self.level-1, self.level-L-1, -1):
                                    diff = numpy.amax(abs(diff), axis = diff_axis)  
                                
                                error_array = numpy.ma.array(diff, mask = coarse_nodata).filled(self.NoDataValue)
                                if error_array.shape == ():#Avoid 0-shape arrays
                                    error_array = error_array.reshape(1,) 
                                
                        elif len (h5file.listNodes(fine_where, "Array"))>1:
                            #WARNING THIS OPTION IS NOT WORKING AS i HAVE NOT IMPLEMENTED THE BASE LAYER IN THIS CASE
                            coarse_array[int(array.name)] = numpy.ma.array(coarse_data, mask = coarse_nodata).filled(self.NoDataValue) 
                            if error:
                                if loop:
                                    inter = loop[1]-loop[0]
                                    diff = (np_array - coarse_data.reshape(coarse_data.shape + (1L,)) + (inter)/2.0)%inter - inter/2.0
                                else:
                                    diff = np_array - coarse_data.reshape(coarse_data.shape + (1L,))
                                err = numpy.amax(abs(diff), axis = axis)
                                error_array[int(array.name)] = numpy.ma.array(err, mask = coarse_nodata).filled(self.NoDataValue)
                        else:
                            arcpy.AddWarning('No nodes present in pyr L{0} {1}'.format(L-1, fine_where))
                        
                    #Update the array in the tree and move on to next leaf
                    coarse_table = h5file.createArray(coarse_where, coarse_name, coarse_array, title = ''.join(str(x) for x in (it.multi_index))+'0'*(self.level-L), createparents=True)
                    coarse_table.flush()
                    if error:
                        error_table = h5file.createArray(error_where, error_name, error_array, title = ''.join(str(x) for x in (it.multi_index))+'0'*(self.level-L-1), createparents=True)
                        error_table.flush()
                        
                    it.iternext()
                del it
            print h5file
                
    def VarResTree(self, tolerance = 1e-08, loop = None):
        """Forms a pyramid Tree where values are propogated to higher levels if they are within tolerance, Nodata otherwise
        If loop is true, include the value whicj is equivalent to 0 (ie 7 for HHSM 2pi for radians)
        Works but replaced with sparse 2. May be useful for some purposes"""

        #Open the file            
        with tables.openFile(self.h5, mode = "r+", title = os.path.basename(self.fname)) as h5file:
            
            for L in range(1, self.level):#for each level of the pyramid to be drawn
            
                #determine the size of the .h5 tree and create an iterator.
                arcpy.AddMessage('Calculating variable resolution tolerance: {0} Pyramid Level {1}'.format(tolerance, L))
                if self.level-L - self.agg > 0:
                    group_array = numpy.zeros((self.aperture,)*(self.level-self.agg-L))
                else:
                    group_array = numpy.zeros((1,))
                it = numpy.nditer(group_array, flags=['multi_index'])
                arcpy.AddMessage('Tree size: {0}'.format(group_array.shape))
                del group_array
                                
                #create a blank array
                coarse_array = numpy.array(numpy.zeros((self.aperture,)*min(self.agg, self.level-L)))
                coarse_array.fill(self.NoDataValue)                
                if coarse_array.shape == ():#Avoid 0-shape arrays
                    coarse_array = coarse_array.reshape(1,)
                arcpy.AddMessage('Array size: {0}\n'.format(coarse_array.shape))
                
                #for each leaf in the tree create an array that is the operation output of the corresponding area of the level below
                while not it.finished:
                    #define the file name varaiables                    
                    coarse_where ="/L{0}".format(L) + ''.join('/'+str(x) for x in (it.multi_index)[:-1])
                    if self.level - self.agg - L > 0:
                        fine_where ="/L{0}".format(L-1) + ''.join('/'+str(x) for x in (it.multi_index))
                    else:
                        fine_where ="/L{0}".format(L-1) + ''.join('/'+str(x) for x in (it.multi_index)[:-1])
                    coarse_name = str(it.multi_index[-1:])[1:-2]
                    
                    #Determine the operation results and place them in the coarse array
                    for array in h5file.walkNodes(fine_where, "Array"):
                        A = array.read()
                        #mask out Nodata cells and calculate means
                        axis = len(A.shape) - 1 
                        nodata = numpy.equal(A, self.NoDataValue)
                        coarse_nodata = nodata.any(axis)
                        coarse_dataA = numpy.ma.array(A, mask= nodata).mean(axis)
                        if loop:#closed arithmetc
                            B = numpy.copy(A)
                            B[B>pi] -= loop                                                      
                            coarse_dataB = numpy.ma.array(B, mask= nodata).mean(axis)
                        #mask out cells where fine data is not close to the mean
                        #No direct tool so iterate the result array
                        #Need to find non-iterating method to reduce calc time
                        out_it = numpy.nditer(coarse_dataA, flags=['multi_index'], op_flags=['readwrite'])
                        while not out_it.finished:
                            if not numpy.allclose(out_it[0], A[out_it.multi_index], atol = tolerance):
                                if loop and numpy.allclose(coarse_dataB[out_it.multi_index], B[out_it.multi_index], atol = tolerance):
                                    out_it[0] = (coarse_dataB[out_it.multi_index]+(loop))%(loop)
                                else:
                                     out_it[0] = self.NoDataValue
                            out_it.iternext()
                        del out_it
                        
                        if len (h5file.listNodes(fine_where, "Array"))==1:
                            coarse_array = numpy.ma.array(coarse_dataA, mask = coarse_nodata).filled(self.NoDataValue)
                        elif len (h5file.listNodes(fine_where, "Array"))>1:
                            coarse_array[int(array.name)] = numpy.ma.array(coarse_dataA, mask = coarse_nodata).filled(self.NoDataValue)
                    #Something like this?
#                        for array in h5file.walkNodes(fine_where, "Array"):                            
#                            A = array.read()
#                            #mask out Nodata cells and calculate means
#                            axis = len(A.shape) - 1                            
#                            nodata = numpy.equal(A, self.NoDataValue)
#                            coarse_nodata = nodata.any(axis)
#                            coarse_data = numpy.ma.array(A, mask= nodata).mean(axis)
#                            if loop:#closed arithmetc
#                                B = numpy.copy(A)
#                                B[B>pi] -= 2*pi                                                      
#                                coarse_dataB = numpy.ma.array(B, mask= nodata).mean(axis)
#                               
#                            close = numpy.allclose(coarse_data, A, atol = tolerance)
#                            not_close_data = coarse_nodata * close
#                            coarse_array[int(array.name)] = numpy.ma.array(coarse_data, mask = not_close_data).filled(self.NoDataValue)
                        
                        else:
                            arcpy.AddWarning('No nodes present in pyr L{0} {1}'.format(L-1, fine_where))
                        
                    #Update the array in the tree and move on to next leaf
                    table = h5file.createArray(coarse_where, coarse_name, coarse_array, title = ''.join(str(x) for x in (it.multi_index))+'0'*L, createparents=True)
                    table.flush()
                    it.iternext()
                del it
                
#    def sparse(self, tolerance):
#        '''Scans an error tree for cells that meet the tolerance.  and are not no Data. resulting values are stored on a sparse tree, with no arrays used.
#        This version was intended to perform direct array comparison but has not been completed'''
#        pass
#        #open the data file
#        with tables.openFile(self.h5, mode = "r+", title = os.path.basename(self.fname)) as h5file: 
#            #iterate through the arrays at finest resolution   
#            print h5file
#            for array in h5file.walkNodes("/L0", classname = "Array"):
#                #Find its hierarchy friends
#                arrays = []
#                for L in range(1, self.level):                    
#                    par = ['0'] + array._v_pathname.split('/')[self.agg-self.level:] 
#                    ind = max(0, self.level-self.agg-L)
#                    diff = min(max(0, self.level-self.agg-L),1)
#                    
#                    dex = self.level-self.agg-L+1
#                    if dex < 1:
#                        fin_dex = '-1'
#                    else:
#                        fin_dex = par[dex]
#                    where = '/E{0}/'.format(L) + "/".join(i for i in par[diff : ind])
#                    name = par[ind]
#                    print diff, ind
#                    print where, name, fin_dex
#                    int_dex = int(fin_dex)
#                    if int_dex >= 0:
#                        arrays += [h5file.getNode(where, name).read()[int_dex]]
#                    else:
#                        arrays += [h5file.getNode(where, name).read()]
#                #do comparison 
#                
#                #iterate levels 
#                #store outputs

    def sparse2(self, tolerance):
        '''Scans an error tree for cells that meet the tolerance.  and are not no Data. resulting values are stored on a sparse tree, with no arrays used.'''
        #open the data file
        print "determining sparse"
        st0 = time.time()
        with tables.openFile(self.h5, mode = "r+", title = os.path.basename(self.fname)) as h5file:
            #create the open file object
            op = openHSM(h5file, self)
            #If /S exists delete it
            if op.open_h5.__contains__('/S'):
                op.open_h5.removeNode('/', 'S', recursive = True)
            #Start at coursest E first val is a (1,) shape array
            vals = op._get_vals((0,), self.level, val_type = 'E')
            if vals[0] > tolerance or vals[0] == self.NoDataValue:
                op._toleranceCheck((), self.level-1, tolerance)
                #tolerance check writes to output and performs recursive
            else:
                print 'Some kind of sparse problem'
        st1 = time.time()
        print "Time to calculate sparse: {0}".format(st1-st0)

        
 
    
    def drawSparse(self, geometry = 'POLYGON', name = ''):
        """Creates a single featureclass which includes all values in the sparse array.
        Polygon size and rot (HHSM) is set by level of value
        Draws voronoi tesselations of the centre point."
        name is added to the end of the featureclass name
        **Warning Assumes negative nodata""" 
        
        arcpy.AddMessage('Drawing sparse: {0}_{1}'.format(self.fname, name))
        print 'Drawing sparse: {0}_{1}'.format(self.fname, name)
        t0 = time.time()
        
        #Set environments
        arcpy.env.overwriteOutput = True
        arcpy.env.XYResolution = 0.0001
        arcpy.env.XYTolerance = 0.001
        
        #filenames
        geometry = geometry.lower()
        if geometry == 'polygon':
            template = r"Z:\Jo\Uni\PhD\models\Toolboxes\Templates.gdb\HexTemplate"
        elif geometry == 'point':
            template = r"Z:\Jo\Uni\PhD\models\Toolboxes\Templates.gdb\HexTemplate_Point"
        temp_file = self.fname +'_'
        gdb, fname = os.path.split(self.fname)
        
        #Accumulate the max min values to calc the representation field
        #Set to use l0. Maybe make a parameter
        minimum, maximum = self.MinMax(data_type = 'S')
        
        #check gdb exists if not 
        if not os.path.exists(gdb):
            arcpy.AddError("Draw levels failed\n" + gdb + " does not exist")
            #arcpy.Delete_management(temp_file)
            return

        #prepare the template
        spatial_ref = arcpy.SpatialReference()
        spatial_ref.loadFromString(self.projection)
        arcpy.Copy_management(template, temp_file)
        arcpy.DefineProjection_management(temp_file , spatial_ref)
        
        #add more fields for the HIP address
        HIP_layers = []
        for i in range(self.level)[::-1]: #order is reversed to match reading direction
            HIP_layers += ['hip_{0}'.format(i)]
            arcpy.AddField_management (temp_file, 'hip_{0}'.format(i), 'LONG', field_is_nullable = True)
            
        #create the feature dataset
        if not arcpy.Exists(self.fname):
            arcpy.CreateFeatureDataset_management (gdb, fname, spatial_ref)
        
        #open the h5file
        with tables.openFile(self.h5, mode = "r") as h5file:
            #create a hip object
            solve = calcHSM(self)
            
            # Create the output feature class
            fc_name = '{0}_sparse_{1}'.format(fname, name)
            fc_fullname = "{0}//{1}".format(self.fname,fc_name)
            arcpy.CreateFeatureclass_management(self.fname, fc_name, geometry, temp_file)
            #np_title = tuple (int(val) for val in array.title)
            
            #open the featureclass for inserting rows
            if geometry == 'polygon':
                cur = arcpy.da.InsertCursor(fc_fullname, ["SHAPE@","Value", "RULEid", "u", "v", ]+HIP_layers)
            elif geometry == 'point':
                cur = arcpy.da.InsertCursor(fc_fullname, ["SHAPE@XY","Value", "RULEid", "u", "v"]+HIP_layers)
            
            #Walk the tree
            for array in h5file.walkNodes('/S', classname =  "Array"):             
                #determine the HIP index
                short_HIP = tuple(int(b) for b in array._v_pathname.split('/')[2:]) 
                HIP = short_HIP + (0,)*(self.level-len(short_HIP))
                # Create the geometry                        
                if geometry == 'polygon':
                    PolyArray = arcpy.Array()
                    pnt = arcpy.Point()
                                            
                    for cnr in solve.xy_corners(HIP, self.level-len(short_HIP))[1:]:#don't process centre
                        pnt.X, pnt.Y = cnr
                        PolyArray.add(pnt)   
                    shape = arcpy.Polygon(PolyArray)
                
                elif geometry == 'point':
                    shape = solve.IP2xy(HIP)
                
                #calc the representation field
                print self.NoDataValue, array[0]
                if self._NoDataCheck(array[0]):
                    #print "True"
                    RuleID = 33
                    val = self.NoDataValue
                else:
                    #print "false"
                    try:
                        RuleID = max(1,(array[0]-minimum)*32/(maximum-minimum)) #32 is number of rules in the representation
                        val = array[0] 
                    except:
                        arcpy.AddMessage("Value Rule problem: {0}, Min: {1}, Max: {2}".format(HIP, minimum, maximum))
                        
                #insert and next
                print val, type(array[0]), RuleID, list(HIP)+[0]*(self.level-len(HIP))
                #print
                cur.insertRow([shape, val, RuleID]+[0,0]+list(HIP)+[0]*(self.level-len(HIP)))
            del cur
                    
            arcpy.Delete_management(temp_file)
            t1 = time.time()
            arcpy.AddMessage('Time to draw {0}: {1} seconds'.format(self.fname, t1-t0))
            
                
#    def InterVarTree(self, tolerance = 1e-08, D6 = False):
#        #retrospectively sets to no data areas where there is no match between adjacent levels. The coarser level is changed. if D6 is true the rotation between levels is taken into account
#        #Each value within the level must be within tolernce and there must be agrement within tolerance with the independent calculation at the coarser level. The coarse level calculation is retained.
#        #Open the file 
#        #replaced with sparse2           
#        with tables.openFile(self.h5, mode = "r+", title = os.path.basename(self.fname)) as h5file:
#            
#            #Iterate through the levels
#            for L in range(self.level-1):#Can skip the last level
#                #iterate through the arrays
#                for fine_array in h5file.walkNodes("/L{0}".format(L), "Array"):
#                    print fine_array._v_pathname, fine_array.title
#                    #find coarse array
#                    #This depends on branch geometry one branch or many.
#                    if len (h5file.listNodes(fine_array._v_parent, "Array"))==1:
#                        #The coarse level has the same location
#                        coarse_where = fine_array._v_parent._v_pathname.replace("/L{0}".format(L), "/L{0}".format(L+1))
#                        coarse_name = fine_array.name
#                        coarse_array = h5file.getNode(coarse_where, coarse_name)
#                        np_coarse = coarse_array.read()
#                        np_fine = fine_array.read()
#                        it =  numpy.nditer(np_coarse, flags=['multi_index'], op_flags=['readwrite'])
#                        while not it.finished: 
#                            values = np_fine[it.multi_index]
#                            value = it[0]
#                            it.iternext()
#                    
#                    elif len (h5file.listNodes(fine_array._v_parent, "Array"))>1:
#                        #The coarse level is in a location truncated by one address
#                        coarse_where =  fine_array._v_parent._v_pathname.replace("/L{0}".format(L), "/L{0}".format(L+1))
#                        if self.level - self.agg - L > 0:
#                            coarse_name = '0'
#                        else:
#                            coarse_name = fine_array._v_parent.name
#                        coarse_array = h5file.getNode(coarse_where, coarse_name)
#                        np_coarse = coarse_array.read()
#                        np_fine = fine_array.read()
#                        it =  numpy.nditer(np_coarse, flags=['multi_index'], op_flags=['readwrite'])
#                        while not it.finished: 
#                            values = np_fine[it.multi_index[:-1]]
#                            value = it[0]
#                            print value, values
#                            it.iternext()   
#                       
#                    print coarse_where, coarse_name
#                    #iterate through the values of the coarse array
#
#        #all close on the apprpriate values of the level below
        
        
#    def CollapseVarTree(self, output_fname):
#        #creates a new HSM at base resolution using the values from the var resolution
#        #ie values are reduplicated
#        #Not completed
#        #Use _collapseSparse
#      
#        #create the output
#        output = self.Copy(output_fname)
#        output.CreateHIP()
#        
#        with tables.openFile(output.h5, mode='r+') as out_file:
#            with tables.openFile(self.h5) as open_file:
#                open_HSM = openHSM(open_file, self)
#                out_HSM = openHSM(out_file, output)
#                for i in range(self.aperture):
#                    for val in open_HSM._LookDown((i,)):
#                        out_HSM._set_vals(val[0], val[1], 0)
#                            
#                            
#            
#        #Start at coarsest level of input
#        #If not no data assign value to corresponding output cells.
#        #If No data drop to the next level down
#        #Go to next level
    
    def _collapseSparse(self, outputname, acc = False):
        """iterates through the sparse values and uses populates the values in base level
        Used in determining errors associated with compression
        Saves the output to output name
        Values not in sparse are set to noData
        If acc = True: The value will be divided by [relative cell area] no divide
        by difference in cellsize (aperture**0.5)."""
        pass
        #Create output
        output = self.Copy(outputname)
        output.CreateHIP()
        #Open input and output
        with tables.openFile(output.h5, mode='r+') as out_file:
            op = openHSM(out_file, output)
            with tables.openFile(self.h5) as in_file:
                #Iterate sparse
                for array in in_file.walkNodes('/S', classname =  "Array"):             
                    #determine the HIP index
                    short_HIP = tuple(int(b) for b in array._v_pathname.split('/')[2:]) 
                    HIP = short_HIP# + (0,)*(self.level-len(short_HIP))
                    val = array[0]
                    if acc == True:
                        val = val/((self.aperture**0.5)**(self.level-len (short_HIP)))
                    #Set values to output
                    op._set_vals(val, HIP, 0)
        return output
        
        
    
                    
    def UpdateValues(self):
        #Applies the HHSM values to the featureclass
        #Looks like a good idea, I'm not sure if this is up to date
        arcpy.AddMessage('Updating Values')
        
        #first need to know the minimum and maximum values for the representation
        minimum, maximum = self.MinMax()
                      
        with tables.openFile(self.h5, mode = "r") as h5file:
            for array in h5file.walkNodes("/L0", "Array"):
                np = array.read()
                #for each table we need to find the associated fc
                fc = "{0}\{1}_{2}".format(self.fname, os.path.basename(self.fname), array.title)
                
                #Identify the HIP fields        
                field_list = arcpy.ListFields(fc)
                short_list = filter(lambda x: "hip" in x, [i.name for i in field_list])#filter out those with hip_ in, may give false positives if the name of the file has hip in it
                short_list.sort(reverse = True)#put them in order
                
                rows = arcpy.da.UpdateCursor (fc , short_list+['value', 'RuleID'])
                #Set the value and RuleID of the fc               
                for row in rows:
                    HIP = tuple(row[:-2]) #evaluate the key
                    #Need to ignore the leading 0 if present, created by projection of HIP where level = agg
                    
                    row[-2]  = np[HIP[max(len(HIP)-self.level,self.level-self.agg):]]#ignore the leading 0
                    if row[-2] == self.NoDataValue:
                        row[-1] = 33
                    else:
                        try:
                            row[-1] = max(1,(row[-2] -minimum)*32/(maximum-minimum)) #32 is number of rules in the representation
                        except:
                            arcpy.AddMessage("Value Rule problem: {0}, Min: {1}, Max: {2}".format(row[-2], minimum, maximum))
                    rows.updateRow(row) 
                # Delete cursor and row objects to remove locks on the data 
                del row, rows
                
    def MinMax(self, level = 0, data_type = 'L'):
        """Determins the minimum and maximum values of the array tree, 
        No data values are excluded unless the entire HHSM is nodata, in which case NoData is returned
        ***WARNING Assumes nodata value is negative """
        if data_type in ['L', 'E']:
            mins = []
            maxs = []
            base = "/L{0}".format(level)
            with tables.openFile(self.h5, mode = "r") as h5file:
                for array in h5file.walkNodes(base, "Array"):
                    np = array.read()
                    values = np < self.NoDataValue
                    if values.any():
                        mins += [np[values].min()]
                        maxs += [np[values].max()]
            if len (mins) > 0:
                minimum = min(mins)
                maximum = max(maxs)
            else:
                minimum = self.NoDataValue
                maximum = self.NoDataValue
                
        elif data_type == 'S':
            cmin = float('inf')
            cmax = float('-inf')
            with tables.openFile(self.h5, mode = "r") as h5file:
                for node in h5file.walkNodes("/S", classname = "Array"):
                    if node[0] < self.NoDataValue:
                        cmin = min(cmin, node[0])
                        cmax = max(cmax, node[0])
            if max != float('inf'):
                minimum = cmin
                maximum = cmax
            else:
                minimum = self.NoDataValue
                maximum = self.NoDataValue
                       
        return minimum, maximum
        
    def DrawLevels(self, start = 0, stop = None, zone = [0], geometry = 'point', data = 'L', drawNoData = True):
        """Creates levels of an array for a layer by drawing the geometries.
        Draw all the layers between start and stop.
        zone defines the region of the HHSM to draw. Not implemented
        Draws voronoi tesselations of the centre point.
        Will not draw levels >= stop value
        If drawNoData: NoData cells will be drawn""" 
        
        arcpy.AddMessage('Drawing Levels: '+self.fname)
        t0 = time.time()
        
        #Set environments
        arcpy.env.overwriteOutput = True
        arcpy.env.XYResolution = 0.0001
        arcpy.env.XYTolerance = 0.001
        
        #filenames
        geometry = geometry.lower()
        if geometry == 'polygon':
            template = r"Z:\Jo\Uni\PhD\models\Toolboxes\Templates.gdb\HexTemplate"
        elif geometry == 'point':
            template = r"Z:\Jo\Uni\PhD\models\Toolboxes\Templates.gdb\HexTemplate_Point"
        temp_file = self.fname +'_'
        gdb, fname = os.path.split(self.fname)
        
        #Accumulate the max min values to calc the representation field
        #Set to use the start level, sometimes it may be preferable to use the l0. Maybe make a parameter
        minimum, maximum = self.MinMax(start)
        
        #if not specified the stop value is the level number. nb 0 is a valid stop value
        if not stop and stop != 0:
            stop = self.level
            
        #If drawing E there is no E0
        if data == 'E':
            start = max(start, 1)
        
        #check gdb exists if not 
        if not os.path.exists(gdb):
            arcpy.AddError("Draw levels failed\n" + gdb + " does not exist")
            #arcpy.Delete_management(temp_file)
            return

        #prepare the template
        spatial_ref = arcpy.SpatialReference()
        spatial_ref.loadFromString(self.projection)
        arcpy.Copy_management(template, temp_file)
        arcpy.DefineProjection_management(temp_file , spatial_ref)
        
        #add more fields for the HIP address
        HIP_layers = []
        for i in range(self.level)[::-1]: #order is reversed to match reading direction
            HIP_layers += ['hip_{0}'.format(i)]
            arcpy.AddField_management (temp_file, 'hip_{0}'.format(i), 'LONG', field_is_nullable = True)
            
        #create the feature dataset
        if not arcpy.Exists(self.fname):
            arcpy.CreateFeatureDataset_management (gdb, fname, spatial_ref)
        
        #open the h5file
        with tables.openFile(self.h5, mode = "r") as h5file:
            #create a calcHSM object
            solve = calcHSM(self)
            
            for L in range(start, stop):
                arcpy.AddMessage('Drawing level: {0}{1}'.format(data, L))
                
                #Walk the tree
                for array in h5file.walkNodes('/{0}{1}'.format(data, L), "Array"):
                    #self.agg is the largest permitted layer. Therefore if the output > than the agg size multiple output featureclasses are needed to store the level. The  structure of the pyramid level defines the fc structure.
                    # Create the output feature class
                    fc_name = '{0}_{1}{2}_{3}'.format(fname, data, L, array.title)
                    fc_fullname = "{0}//{1}".format(self.fname,fc_name)
                    arcpy.CreateFeatureclass_management(self.fname, fc_name, geometry, temp_file)
                    np_title = tuple (int(val) for val in array.title)
                    
                    #open the featureclass for inserting rows
                    if geometry == 'polygon':
                        cur = arcpy.da.InsertCursor(fc_fullname, ["SHAPE@","Value", "RULEid", "u", "v", ]+HIP_layers)
                    elif geometry == 'point':
                        cur = arcpy.da.InsertCursor(fc_fullname, ["SHAPE@XY","Value", "RULEid", "u", "v"]+HIP_layers)
                    
                    #walk the array
                    np_array = array.read()                                       
                    it = numpy.nditer(np_array, flags=['multi_index'])
                    while not it.finished:
                        #IgnoreNoData
                        if drawNoData == True or it[0] != self.NoDataValue:
                            #determine the HIP index
                            HIP = (np_title[:-self.agg]+it.multi_index+(0,)*(L))[-self.level:]
                            # Create the geometry                        
                            if geometry == 'polygon':
                                PolyArray = arcpy.Array()
                                pnt = arcpy.Point()
                                                        
                                for cnr in solve.xy_corners(HIP, L)[1:]:#don't process centre
                                    pnt.X, pnt.Y = cnr
                                    #print cnr
                                    PolyArray.add(pnt)   
                                shape = arcpy.Polygon(PolyArray)
                            
                            elif geometry == 'point':
                                shape = solve.IP2xy(HIP)
                            
                            #calc the representation field
                            if it[0] == self.NoDataValue:
                                RuleID = 33
                            else:
                                try:
                                    RuleID = max(1,(it[0]-minimum)*32/(maximum-minimum)) #32 is number of rules in the representation
                                except:
                                    arcpy.AddMessage("Value Rule problem: {0}, Min: {1}, Max: {2}".format(HIP, minimum, maximum))
                                    
                            #insert and next
                            #print it[0], type(it[0])
                            cur.insertRow([shape, it[0], RuleID]+list(solve.IP2xy(it.multi_index))+list(HIP))
                        it.iternext()
                    del cur
                    
            arcpy.Delete_management(temp_file)
            t1 = time.time()
            arcpy.AddMessage('Time to draw {0}: {1} seconds'.format(self.fname, t1-t0))
            
    
            
#    def Delete(self):
    #Depreciated use arihmetic
#        #Deletes the associated h5.file
#        #should be used instead of manual methods to avoid redundent file accumulation
##Arguably should delete Feature dataset as well (if exists). Check names       
#        os.remove(self.h5)
#        
#    def Subtract(self, sub_ob, outFname, start_level = 0, stop_level = 1, loop = None):
#        """
#            subtracts RHSM_ob from self
#            returns new object saved as outFname
#            self and sub_op must be same level, agg, origin, cellsize...
#            loop: list of floats [lower, upper]
#        """
#        
#                #create the output RHSM
#        out = copy.deepcopy(self)
#        out.h5 = outFname+'.h5'
#        out.fname = outFname
#        #delete h5 file it currently has the copied values.
#        if os.path.exists(out.h5):
#            os.remove(out.h5)
#        out.CreateHIP()
#        if stop_level > 1:
#            out.PyrTree()#Creates blank files
#            
#        #open the input data
#        with tables.openFile(self.h5, mode = "r") as in_file:
#            #open the subtract data
#            with tables.openFile(sub_ob.h5, mode = "r") as sub_file:
#                #open the output data
#                with tables.openFile(out.h5, mode = "r+") as out_file:
#                    #go through the levels
#                    for level in range(start_level, stop_level):
#                        base = "/L{0}".format(level)
#                        #walk through the arrays on the specified level           
#                        for in_array in in_file.walkNodes(base, "Array"):
#                            #create an input numpy array
#                            in_np = in_array.read()
#                            #find its path name get the subtract array with the same name
#                            sub_array = sub_file.getNode(in_array._v_pathname)
#                            sub_np = sub_array.read()
#                            #get the output array with the same name
#                            out_array = out_file.getNode(in_array._v_pathname)
#                            #mask out No Data cells and pits
#                            in_np_shape = in_np.shape
#                            #pits = in_np == -1#Not worried about this yet
#                            #pits = pits.reshape(in_np_shape)
#                            in_nodata = in_np == self.NoDataValue
#                            in_nodata = in_nodata.reshape(in_np_shape)
#                            sub_nodata = sub_np == sub_ob.NoDataValue
#                            sub_nodata = sub_nodata.reshape(in_np_shape)
#                            #Apply the subtraction
#                            out_np = in_np - sub_np
#                            #apply loop
#                            if loop:
#                                topLoop = out_np > (loop[1]-loop[0])/2
#                                topLoop = topLoop.reshape(in_np_shape)
#                                bottomLoop = out_np < -(loop[1]-loop[0])/2
#                                bottomLoop = bottomLoop.reshape(in_np_shape)
#                                out_np = numpy.where(topLoop, out_np-2*pi, out_np)
#                                out_np = numpy.where(bottomLoop, out_np+2*pi, out_np)
#                            #apply masks
#                            out_np[in_nodata] = self.NoDataValue
#                            out_np[sub_nodata] = self.NoDataValue
#                            out_np[out_np == numpy.inf] = self.NoDataValue
#                            #set to output data
#                            out_array[:]=out_np
#                            out_array.flush()
#        return out
                    
    def arithmetic(self, end_ob, outFname, operation = '+', start_level = 0, stop_level = 1, loop = None):
        """
            Perfoms self operation en_ob
            operation must be a string representation of an operation that can be performed on numpy arrays
            ie "+", "-", "*", "/", "**"...
            returns new object saved as outFname
            self and div_op must be same level, agg, origin, cellsize...
        """
        
                #create the output RHSM
        out = copy.deepcopy(self)
        out.h5 = outFname+'.h5'
        out.fname = outFname
        #delete h5 file it currently has the copied values.
        if os.path.exists(out.h5):
            os.remove(out.h5)
        out.CreateHIP()
        if stop_level > 1:
            out.PyrTree()#Creates blank files
            
        #open the input data
        with tables.openFile(self.h5, mode = "r") as in_file:
            #open the subtract data
            with tables.openFile(end_ob.h5, mode = "r") as sub_file:
                #open the output data
                with tables.openFile(out.h5, mode = "r+") as out_file:
                    #go through the levels
                    for level in range(start_level, stop_level):
                        base = "/L{0}".format(level)
                        #walk through the arrays on the specified level           
                        for in_array in in_file.walkNodes(base, "Array"):
                            #create an input numpy array
                            in_np = in_array.read()
                            #find its path name get the subtract array with the same name
                            sub_array = sub_file.getNode(in_array._v_pathname)
                            sub_np = sub_array.read()
                            #get the output array with the same name
                            out_array = out_file.getNode(in_array._v_pathname)
                            #mask out No Data cells and pits
                            in_np_shape = in_np.shape
                            #pits = in_np == -1#Not worried about this yet
                            #pits = pits.reshape(in_np_shape)
                            in_nodata = in_np == self.NoDataValue
                            in_nodata = in_nodata.reshape(in_np_shape)
                            sub_nodata = sub_np == end_ob.NoDataValue
                            sub_nodata = sub_nodata.reshape(in_np_shape)
                            #Apply the division
                            out_np = eval("in_np {0} sub_np".format(operation))
                            #apply loop
                            if loop:
                                topLoop = out_np > (loop[1]-loop[0])/2
                                topLoop = topLoop.reshape(in_np_shape)
                                bottomLoop = out_np < -(loop[1]-loop[0])/2
                                bottomLoop = bottomLoop.reshape(in_np_shape)
                                out_np = numpy.where(topLoop, out_np-2*pi, out_np)
                                out_np = numpy.where(bottomLoop, out_np+2*pi, out_np)
                            #apply masks
                            out_np[out_np == numpy.inf] = self.NoDataValue
                            out_np[in_nodata] = self.NoDataValue
                            out_np[sub_nodata] = self.NoDataValue
                            #set to output data
                            out_array[:]=out_np
                            out_array.flush()
        return out
                        
        
        
    def _dinf2(self, hood):
        #direction to the reference plane (assumed to be x,y axis at some lower z value)
        ref_vector = (0, 0, -1)
        faces = numpy.array([[0.0, 0.0]]*(self.aperture-1))
        for i in range (self.aperture - 1):
            #find the three points of triangle, translation is irrelavent (x, y, z)
            p0 = numpy.array((0.0, 0.0, hood[0]), dtype = float)
            xy = self._convIP2xy((i + 1,)).flatten()
            p1 =  numpy.array((xy[0], xy[1], hood[i + 1]), dtype = float)
            xy = self._convIP2xy(((i + 1)%(self.aperture-1)+1,)).flatten()
            p2 = numpy.array((xy[0], xy[1], hood[(i + 1)%(self.aperture-1)+1]), dtype = float)

            #1. determine unit normal of plane
            v1 = p1 - p0
            v2 = p2 - p0
            n = numpy.cross(v1, v2)
            un = n/(numpy.sqrt((n*n).sum()))
            #2. Project the reference vector onto the plane          
            vg = ref_vector-(numpy.dot(ref_vector, un))*un
            
            #3. determine direction and slope
            ri_p = atan2(vg[1], vg[0]) #clockwise from east
            ri = (2*pi+ri_p)%(2*pi)#make angle always positive
            si = vg[2]/(vg[0]**2 + vg[1]**2)**0.5
            
            #4. Adjust to edges if necessary
            #determine smallest difference between angles
            ri1 = atan2(v1[1], v1[0])
            ri1_n = (2*pi+ri1)%(2*pi)
            ri1_diff_n = min((2*pi) - abs(ri - ri1_n), abs(ri - ri1_n))
            ri2 = atan2(v2[1], v2[0])
            ri2_n = (2*pi+ri2)%(2*pi)
            ri2_diff_n = min((2*pi) - abs(ri - ri2_n), abs(ri - ri2_n))
            #If the angle is outside the wedge formed between angles
            if max(ri1_diff_n, ri2_diff_n) >= 2*pi/(self.aperture-1):
                #Assign the values of the nearest angle
                if ri1_diff_n <= ri2_diff_n:#arbitrary assignment if equal
                    ri = ri1_n
                    si = v1[2]/float((v1[0]**2+v1[1]**2)**0.5)
                elif ri2_diff_n < ri1_diff_n:
                    ri = ri2_n
                    si = v2[2]/float((v2[0]**2+v2[1]**2)**0.5)
            
            faces[i] = [ri, si]   
        #assign the  direction    
        dir_index = list(faces[:,1]).index(min(faces[:,1]))#this line identifies (first) index of the steepest slope
        if faces[dir_index][1] > 0:
            #mark as a sink
            direction = -1
            
        else:
            direction = faces[dir_index][0]
        return direction
        
    def _dinf3(self, hood):
        """calculate rec dinf in a rooks case."""
        #direction to the reference plane (assumed to be x,y axis at some lower z value)
        ref_vector = (0, 0, -1)
        faces = numpy.array([[0.0, 0.0]]*(self.aperture-1))
        for i in range (0,self.aperture - 1, 2):
            #find the three points of triangle, translation is irrelavent (x, y, z)
            p0 = numpy.array((0.0, 0.0, hood[0]), dtype = float)
            xy = self._convIP2xy((i + 1,)).flatten()
            p1 =  numpy.array((xy[0], xy[1], hood[i + 1]), dtype = float)
            xy = self._convIP2xy(((i + 3)%(self.aperture-1),)).flatten()
            p2 = numpy.array((xy[0], xy[1], hood[(i + 3)%(self.aperture-1)]), dtype = float)

            #1. determine unit normal of plain
            v1 = p1 - p0
            v2 = p2 - p0
            n = numpy.cross(v1, v2)
            un = n/(numpy.sqrt((n*n).sum()))
            #2. Project the reference vector onto the plain          
            vg = ref_vector-(numpy.dot(ref_vector, un))*un
            
            #3. determine direction and slope
            ri_p = atan2(vg[1], vg[0]) #clockwise from east
            ri = (2*pi+ri_p)%(2*pi)#make angle always positive
            si = vg[2]/(vg[0]**2 + vg[1]**2)**0.5
            
            #4. Adjust to edges if necessary
            #determine smallest difference between angles
            ri1 = atan2(v1[1], v1[0])
            ri1_n = (2*pi+ri1)%(2*pi)
            ri1_diff_n = min((2*pi) - abs(ri - ri1_n), abs(ri - ri1_n))
            ri2 = atan2(v2[1], v2[0])
            ri2_n = (2*pi+ri2)%(2*pi)
            ri2_diff_n = min((2*pi) - abs(ri - ri2_n), abs(ri - ri2_n))
            #If the angle is outside the wedge formed between angles
            if max(ri1_diff_n, ri2_diff_n) >= 2*pi/((self.aperture-1)/2):
                #Assign the values of the nearest angle
                if ri1_diff_n <= ri2_diff_n:#arbitrary assignment if equal
                    ri = ri1_n
                    si = v1[2]/float((v1[0]**2+v1[1]**2)**0.5)
                elif ri2_diff_n < ri1_diff_n:
                    ri = ri2_n
                    si = v2[2]/float((v2[0]**2+v2[1]**2)**0.5)
            
            faces[i] = [ri, si]   
        #assign the  direction    
        dir_index = list(faces[:,1]).index(min(faces[:,1]))#this line identifies (first) index of the steepest slope
        if faces[dir_index][1] > 0:
            #mark as a sink
            direction = -1
            
        else:
            direction = faces[dir_index][0]
        return direction
            

        
        
class HHSM(HSM):
    """Extends HSM for Hexagonal specific features"""
    
    #Class variables
    aperture = 7 #tree bredth
    #hip to xy
    rotations={0:numpy.array([[0,0],[0,0]]), 1: numpy.array([[1,0],[0,1]]), 2: numpy.array([[0.5,-3**0.5/2.0],[3**0.5/2.0,0.5]]), 3: numpy.array([[-0.5,-3**0.5/2.0],[3**0.5/2.0,-0.5]]), 4: numpy.array([[-1,0],[0,-1]]), 5: numpy.array([[-0.5,3**0.5/2.0],[-3**0.5/2.0,-0.5]]), 6: numpy.array([[0.5,3**0.5/2.0],[-3**0.5/2.0,0.5]])}
    b = numpy.array([[2, -3**0.5],[3**0.5, 2]])
    c = numpy.array([[1],[0]])
    
    #xy corners
    cnr = numpy.array([[0,-1.0/3,1.0/3,2.0/3,1.0/3,-1.0/3,-2.0/3],[0,2.0/3,1.0/3,-1.0/3,-2.0/3,-1.0/3,1.0/3]])
    pyr_rot =  numpy.array([[1, -2],[2, 3]])
        
    #hip to uv acute
    uvA_rotations={0:numpy.array([[0,0],[0,0]]), 1: numpy.array([[1,0],[0,1]]), 2: numpy.array([[0,-1],[1,1]]), 3: numpy.array([[-1,-1],[1,0]]), 4: numpy.array([[-1,0],[0,-1]]), 5: numpy.array([[0,1],[-1,-1]]), 6: numpy.array([[1,1],[-1,0]])}
    uvA_b = numpy.array([[1, -2],[2, 3]])
    uvA_c = numpy.array([[1],[0]])
    
    #hip to uv obtuse
    uvO_rotations={0:numpy.array([[0,0],[0,0]]), 1: numpy.array([[1,0],[0,1]]), 2: numpy.array([[1,-1],[1,0]]), 3: numpy.array([[0,-1],[1,-1]]), 4: numpy.array([[-1,0],[0,-1]]), 5: numpy.array([[-1,1],[-1,0]]), 6: numpy.array([[0,1],[-1,1]])}
    uvO_b = numpy.array([[3, -2],[2, 1]])
    uvO_c = numpy.array([[1],[0]])
    
    #uv acute to xy
    uvA2xy_rot = numpy.array([[1,0.5],[0,3**0.5/2]])
    
    #hip add/subtract
    #values reversed
    IPadd_table = numpy.array([[(0,0),(1,0),(2,0),(3,0),(4,0),(5,0),(6,0)],
                                [(1,0),(3,6),(5,1),(2,0),(0,0),(6,0),(4,6)],
                                 [(2,0),(5,1),(4,1),(6,2),(3,0),(0,0),(1,0)],
                                  [(3,0),(2,0),(6,2),(5,2),(1,3),(4,0),(0,0)],
                                   [(4,0),(0,0),(3,0),(1,3),(6,3),(2,4),(5,0)],
                                    [(5,0),(6,0),(0,0),(4,0),(2,4),(1,4),(3,5)],
                                     [(6,0),(4,6),(1,0),(0,0),(5,0),(3,5),(2,5)]])
    IPnegation_table = [0,4,5,6,1,2,3]
    
    #Nieghbourhood operations
    nieghbour_dist = numpy.array([1, 1, 1, 1, 1, 1, 1])#First value is really 0 but this would create 0 div error for drop calculations
    
    #Dinf
    Dinf_table = numpy.array([[0, 0, 0, 0, 0, 0],
                              [1, 2, 2, 4, 4, 7]])
                              
    #Dinf
    e0 = [0, 0, 0, 0, 0, 0]
    e1 = [1, 2, 3, 4, 5, 6]
    e2 = [2, 3, 4, 5, 6, 1]
    ac = [0, 1, 2, 3, 4, 5]
    
            
    def _uvAcute2xy(self, uv, origin):
        #Input: tuple (u, v)
        #Output: tuple (x, y)
        val = numpy.array([[uv[0]],[uv[1]]])
        result = numpy.dot(self.uvA2xy_rot, val)*self.cellsize+origin
        if self.rotation:
            result = numpy.dot(self.rot, result)                   
        return tuple(result.flatten())
                
    def _HIP2uvObtuse(self, hip):
        #converts from HIP coord to uv. in the obtuse axis form input is a tuple of num level values
        result = numpy.array([0])
        for i, val in enumerate(hip[::-1]):
            result = numpy.add(result,  numpy.dot(numpy.dot(self.uvO_rotations[val], numpy.linalg.matrix_power(self.uvO_b, i)), self.uvO_c).flatten())
        return  tuple(result)
      
    def _HIP2uvAcute(self, hip):
        #input hip as tuple
        #output uv acute as tuple
        #Does not apply rotation (assumes u axis base vector is equivalent to the hip base vector)
        result = numpy.array([0])
        for i, val in enumerate(hip[::-1]):
            result = numpy.add(result, numpy.dot(numpy.dot(self.uvA_rotations[val], numpy.linalg.matrix_power(self.uvA_b, i)),self.uvA_c).flatten())
        return  tuple(result)
        
    def _IP2xy(self, result, i, val):
        result = numpy.add(result, numpy.dot(numpy.dot(self.rotations[val], numpy.linalg.matrix_power(self.b, i)),self.c))
        return result
        
        
    def _calcCorners(self, centre_ip, pyr, origin):
        #Goes through uvAcute. Not yet sure if corners can be defined in HIP
        u, v = self._HIP2uvAcute(centre_ip)
        centre = numpy.array([[u],[v]])
        corners = numpy.add(centre, numpy.dot (numpy.linalg.matrix_power (self.pyr_rot, pyr), self.cnr)).transpose()
        for i, corner  in enumerate(corners):           
            corners[i] = self._uvAcute2xy(corner, origin)
        return corners
        
    def _flip2closest(self, from_level = 0):
        #this internal function changes the values of the pyramid levels of a hex D6 to the nearest 
        #equivalent direction on the from_level
        
        #open the h5 file
        with tables.openFile(self.h5, mode = "r+") as h5file:      
            #iterate levels
            for group in h5file.walkGroups("/"):
                if group._v_name not in ["/", "L0"]:#ignore root and l0
                    pyr_level = int(sub("[^0-9]", "", group._v_name))
                    #iterate arrays
                    for array in h5file.listNodes(group, classname='Array'):
                        #iterate values
                        np = array.read()
                        it =  numpy.nditer(np, flags=['multi_index'], op_flags=['readwrite'])
                        while not it.finished:
                            if it[0] != self.NoDataValue:
                                #change to closest
                                level_rot = (pyr_level-from_level)*atan(3**0.5/2)/atan(3**0.5)
                                closest = int(round((it[0] - level_rot)%6))
                                if closest == 0:
                                    closest = 6
                                #print it[0], closest
                                it[0] = closest
                            it.iternext()
                        array[:]=np
                        array.flush()

    def _flipDinf(self, from_level = 0, to = 'base'):
        #this internal function changes the values of the pyramid levels of a hex Dinf 
        #So that the new direction represents the direction on the cell's LoD that matches the direction on the base LoD.
        #to = 'base' will convert LoD directions to the base direction (subtract)
        #to = 'LoD' will convert base directions to the LoD direction (add)
        #Assumes dinf is in radians (sinks = -1) ie run this before Dinf2HSM
        
        #**I suspect this will not work when agg<level
        
        #open the h5 file
        print "Flipping levels to {0}".format(to)
        with tables.openFile(self.h5, mode = "r+") as h5file:      
            #iterate levels
            for trunk in h5file.iterNodes("/"):
                if trunk._v_name[0] == "L" and trunk._v_name != "L0":#ignore root, S, l0 and E
                    pyr_level = int(sub("[^0-9]", "", trunk._v_name))
                    #print trunk._v_name
                    #level_rot = (pyr_level-from_level)*atan(3**0.5/2)/atan(3**0.5)#this would be for RHSM directions
                    level_rot = (pyr_level-from_level)*atan((3**0.5/2))
                    if to == 'base':
                        level_rot = -level_rot
                    #print level_rot
                    #iterate arrays
                    for array in h5file.walkNodes(trunk, classname='Array'):
                        #iterate values
                        np = array.read()
                        np_shape = np.shape
                        #mask out No Data cells and pits
                        pits = np == -1
                        pits = pits.reshape(np_shape)
                        nodatas = np == self.NoDataValue
                        nodatas = nodatas.reshape(np_shape)
                        np = (np - level_rot)%(2*pi)
                        np[pits] = -1 #Was there a reason I had 0 previously?
                        np[nodatas] = self.NoDataValue
                        array[:]=np
                        array.flush()  

                        
                        
#    def dinf2(self, hood):
#        #direction to the reference plane (assumed to be x,y axis at some lower z value)
#        ref_vector = (0, 0, -1)
#        faces = numpy.array([[0.0, 0.0]]*(self.aperture-1))
#        for i in range (self.aperture - 1):
#            #find the three points of triangle, translation is irrelavent (x, y, z)
#            p0 = numpy.array((0.0, 0.0, hood[0]), dtype = float)
#            xy = self._convIP2xy((i + 1,)).flatten()
#            p1 =  numpy.array((xy[0], xy[1], hood[i + 1]), dtype = float)
#            xy = self._convIP2xy(((i + 1)%(self.aperture-1)+1,)).flatten()
#            p2 = numpy.array((xy[0], xy[1], hood[(i + 1)%(self.aperture-1)+1]), dtype = float)
#
#            #1. determine unit normal of plain
#            v1 = p1 - p0
#            v2 = p2 - p0
#            n = numpy.cross(v1, v2)
#            un = n/(numpy.sqrt((n*n).sum()))
#            
#            #2. Project the reference vector onto the plain          
#            vg = ref_vector-(numpy.dot(ref_vector, un))*un
#            
#            #3. determine direction and slope
#            ri_p = atan2(vg[1], vg[0]) #clockwise from east
#            ri = (2*pi+ri_p)%(2*pi)#make angle always positive
#            si = vg[2]/(vg[0]**2 + vg[1]**2)**0.5
#            
#            #4. Adjust to edges if necessary
#            #determine smallest difference between angles
#            ri1 = atan2(v1[1], v1[0])
#            ri1_n = (2*pi+ri1)%(2*pi)
#            ri1_diff_n = min((2*pi) - abs(ri - ri1_n), abs(ri - ri1_n))
#            ri2 = atan2(v2[1], v2[0])
#            ri2_n = (2*pi+ri2)%(2*pi)
#            ri2_diff_n = min((2*pi) - abs(ri - ri2_n), abs(ri - ri2_n))
#            #If the angle is outside the wedge formed between angles
#            if max(ri1_diff_n, ri2_diff_n) > 2*pi/(self.aperture-1):
#                #Assign the values of the nearest angle
#                if ri1_diff_n <= ri2_diff_n:#arbitrary assignment if equal
#                    ri = ri1_n
#                    si = v1[2]/float((v1[0]**2+v1[1]**2)**0.5)
#                elif ri2_diff_n < ri1_diff_n:
#                    ri = ri2_n
#                    si = v2[2]/float((v2[0]**2+v2[1]**2)**0.5)
#            
#            faces[i] = [ri, si]   
#        #assign the  direction    
#        dir_index = list(faces[:,1]).index(min(faces[:,1]))#this line identifies (first) index of the steepest slope
#        if faces[dir_index][1] > 0:
#            #mark as a sink
#            direction = 0
#        else:
#            #assign as HSM equivalent by angle
#            direction = faces[dir_index][0]/float(2*pi/(self.aperture-1)) + 1
#            #or assign by linear ratio
#            #or assign by linear projection
#        return direction
#        
#                        
#    def dinf(self, hood):
#        #calculates the dinf direction of the hood in vector equivalents (ie ranges from 1 to aperture-1)
#        #calculate the directions and slopes
#        faces = numpy.array([[0.0, 0.0]]*(self.aperture-1))
#        for i in range (self.aperture-1):
#            s1i = self.s1(hood[self.e0[i]], hood[self.e1[i]], hood[self.e2[i]], self.cellsize)
#            s2i = self.s2(hood[self.e1[i]], hood[self.e2[i]], self.cellsize)
#            ri = atan2(s2i, s1i)
#            si = (s1i**2 + s2i**2)**0.5
#            #print 'ri: {0}, si: {1}'.format(ri, si)
#            if ri < -pi/6:
#                ri = -pi/6
#                si = (hood[self.e0[i]] - hood[self.e1[i]])/float(self.cellsize)
#            elif ri > atan(1/3.0**0.5):
#                ri = pi/6
#                si = (hood[self.e0[i]] - hood[self.e2[i]])/float(self.cellsize)
#            #print 'ri new: {0}, si new: {1}'.format(ri, si)
#            faces[i] = [ri, si]
#        dir_index = list(faces[:,1]).index(max(faces[:,1]))#this line identifies (first) index of the steepest slope
#        if faces[dir_index][1] < 0:
#            direction = 0
#        else:            
#            #dir_inf = faces[dir_index][0] + self.ac[dir_index]*pi/3.0 + pi/6.0
#            #direction = dir_inf/(pi/3.0) + 1 #convert to HSM
#            #direction = 3*faces[dir_index][0]/pi + self.ac[dir_index] + 1.5
#            direction = 3**0.5/2.0*tan(faces[dir_index][0]) + self.ac[dir_index] + 1.5
#            #print 3**0.5*tan(faces[dir_index][0]), direction
#        return direction
#        
#        
#    def s1(self, e0i, e1i, e2i, di):
#        #used to determine the slope of side one of the dinf facet
#        result = (e0i - (e1i+e2i)/2)/ (di*3**0.5/2.0)
#        #print 's1: {0}'.format(result)
#        return result
#        
#        
#    def s2(self, e1i, e2i, di):
#        #used to determine the slope of one side two of the dinf facet
#        result = (e1i - e2i)/float(di)
#        #print 's2: {0}'.format(result)
#        return result

    def valuesFrom_UVacute(self, UV_array, U0, V0, limit=None):
        """Takes the values from the UVarray and populates the h5 file
        UV_array is a numpy array with UV indexing.
        U0, V0 are the col, row of the pixel that is the origin of in UV
        No allowance for cell size
        recently moved rom HSM and not checked if working
        """
        
        #Iterate the HSM
        with tables.openFile(self.h5, mode = "r+") as h5file:
            #create a conversion object
            for array in h5file.walkNodes("/L0", "Array"):
                np = array.read()
                it =  numpy.nditer(np, flags=['multi_index'], op_flags=['readwrite'])
                while not it.finished:
                    HIP = tuple (int(val) for val in array.title[:-self.agg])+it.multi_index
                    (u, v) = self._HIP2uvAcute(HIP)
                    #Trim to limit if required
                    #Need to add limit from array size
                    if limit:
                        (x, y) = self._convIP2xy(HIP)
                        if eval(limit) and 0 <= u+U0 < UV_array.shape[0] and 0 <= v+V0 < UV_array.shape[1]:
                            ucell, vcell  = int(u+U0), int(-v+V0)
                            it[0]  = UV_array[vcell, ucell]
                        else:
                            it[0] = self.NoDataValue
                    #Otherwise limit to extent of input array
                    else:
                        if not 0 <= u+U0 < UV_array.shape[0] or not 0 <= v+V0 < UV_array.shape[1]:                            
                            it[0] = self.NoDataValue
                        else:
                            ucell, vcell  = int(u+U0), int(-v+V0)
                            it[0]  = UV_array[vcell, ucell]
                    it.iternext()
                
                array[:]=np
                array.flush() 
        
        
#    def ViewLevels(self, start = 0, stop = None, zone = None):
#        """Creates featureclasses to represent levels of an HHSM between start and stop
#        avoids drawing geometries by redefining the projection on default arrays
#        The featureclass are redefined with a special projection which causes them to appear in the correct position
#        pyr tree must exist for the levels to be viewed
#        not working -Appears to have been replaced with ProjLevels"""
#        arcpy.AddMessage('Viewing Levels: '+self.fname)
#        #Set environments
#        arcpy.env.overwriteOutput = True
#        
#        #filenames
#        gdb, fname = os.path.split(self.fname)
#        
#        #Accumulate the max min values to calc the representation field
#        minimum, maximum = self.MinMax()
#        
#        #Create a Spatial reference object
#        spatial_ref = arcpy.SpatialReference()
#        spatial_ref.loadFromString(self.projection)
#        
#        #if not specified the stop value is the level number
#        if not stop:
#            stop = self.level
#        
#        #check gdb exists if not terminate
#        if not os.path.exists(gdb):
#            arcpy.AddError("Draw levels failed\n" + gdb + " does not exist")
#            #arcpy.Delete_management(temp_file)
#            return
#        #open the h5file
#        with tables.openFile(self.h5, mode = "r") as h5file:
#            calc = calcHSM(self)           
#                        
#            for L in range(start, stop+1):
#                arcpy.AddMessage('Drawing level: L{0}'.format(L))
#                
#                #Walk the tree
#                for array in h5file.walkNodes('/L{0}'.format(L), "Array"):
#                    #self.agg is the largest permitted layer. Therefore if the output > than the agg size multiple output featureclasses are needed to store the level. The  structure of the pyramid level defines the fc structure.
#                    np_array = np_array = array.read()
#                    
#                    # Create the output feature class
#                    fc_name = '{0}_L{1}_{2}_'.format(fname, L, array.title)
#                    fc_fullname = "{0}\{1}".format(gdb, fc_name)
##                    parts += [fc_fullname]
#                    
#                    #find the appropriate template                   
#                    template = "D:\Uni\PhD\models\Toolboxes\Templates.gdb\L{0}_HIP".format(len(np_array.shape))
#                    #rotate and scale
#                    arcpy.AddMessage(tuple(array.title))
#                    e_shift, n_shift  = calc.hip2xy(tuple (int(val) for val in array.title))
#                    scale = self.cellsize*7**(0.5*L)
#                    rotation = degrees(atan(3**.5/2))*L
#                    #need to Identify the latitude and longitude of the origin in the projection of the template projection. Can't remember why?
#                    long_centre, lat_centre = 0, 0
#                    shift_projection = "PROJCS['Rotation_Projection',GEOGCS['GCS_WGS_1984',DATUM['D_WGS_1984',SPHEROID['WGS_1984',6378137.0,298.257223563]],PRIMEM['Greenwich',0.0],UNIT['Degree',0.0174532925199433]],PROJECTION['Local'],PARAMETER['False_Easting',{0}],PARAMETER['False_Northing',{1}],PARAMETER['Scale_Factor',{2}],PARAMETER['Azimuth',{3}],PARAMETER['Longitude_Of_Center',{4}],PARAMETER['Latitude_Of_Center',{5}],UNIT['Meter',1.0]]".format(e_shift, n_shift, scale, rotation, long_centre, lat_centre)
#                    
#                    #translate by project and redefine
#                    arcpy.Project_management(template, fc_fullname, shift_projection) 
#                    arcpy.DefineProjection_management(fc_fullname, spatial_ref)                    
#                    
##                    #translate by copy then redefining the projection
##                    arcpy.Copy_management(template, fc_fullname) 
##                    arcpy.DefineProjection_management(fc_fullname, shift_projection)  
#                    
#                    #Apply value and Field calc the representation
#                    field_list = arcpy.ListFields(fc_fullname) #list of field objects with .name attributes
##                    for field in field_list:
##                        print field.name                          
#
#                    short_list = filter(lambda x: "hip" in x, [i.name for i in field_list])#filter out those with hip_ in, may give false positives if the name of the file has hip_ in it
#                    short_list.sort(reverse = True)#put them in order
#                    
#                    rows = arcpy.da.UpdateCursor (fc_fullname, short_list+['u','value', 'RuleID'])
#                    for row in rows: 
#                        HIP = tuple(row[:-3]) #evaluate the key
#                        row[-2]  = array[HIP]
#                        row[-3] = 4
#                        if row[-2] == self.NoDataValue:
#                            row[-1] = 33
#                        else:
#                            try:
#                                row[-1] = max(1,int((row[-2]-minimum)*32/(maximum-minimum))) #32 is number of rules in the representation
#                            except:
#                                arcpy.AddMessage("Value Rule problem: {0}, Min: {1}, Max: {2}".format(row[-2], minimum, maximum))
#                        rows.updateRow(row)
#                    
#                    # Delete cursor and row objects to remove locks on the data 
#                    del row, rows

    def ProjLevels(self, start = 0, stop = None, zone = None):
        """builds pyrimids by rotating and scaling the defaults
        the rotation is done using a local projection and redefinition. 
        Values are taken from a PyrTree.
        May be worth implementing for other geometries"""
        arcpy.AddMessage('Drawing Levels: '+self.fname)
        
        #Set environments
        arcpy.env.overwriteOutput = True
        
        #filenames
        gdb, fname = os.path.split(self.fname)
        
        #Accumulate the max min values to calc the representation field
        minimum, maximum = self.MinMax()
        
        #if not specified the stop value is the level number
        if not stop:
            stop = self.level
        
        #check gdb exists if not terminate
        if not os.path.exists(gdb):
            arcpy.AddError("Draw levels failed\n" + gdb + " does not exist")
            #arcpy.Delete_management(temp_file)
            return
        
        #Create a Spatial reference object
        spatial_ref = arcpy.SpatialReference()
        spatial_ref.loadFromString(self.projection)
            
        #create the feature dataset
        #exists?
        arcpy.CreateFeatureDataset_management (gdb, fname, spatial_ref)
        
        #Create a list to store the temp outputs
        parts = []
        
        #open the h5file
        with tables.openFile(self.h5, mode = "r") as h5file:            
                        
            for L in range(start, stop+1):
                arcpy.AddMessage('Drawing level: L{0}'.format(L))
                
                #Walk the tree
                for array in h5file.walkNodes('/L{0}'.format(L), "Array"):
                    #self.agg is the largest permitted layer. Therefore if the output > than the agg size multiple output featureclasses are needed to store the level. The  structure of the pyramid level defines the fc structure.
                    np_array = np_array = array.read()
                    
                    # Create the output feature class
                    fc_name = '{0}_L{1}_{2}_'.format(fname, L, array.title)
                    fc_fullname = "{0}\{1}".format(gdb, fc_name)
                    parts += [fc_fullname]
                    
                    #find the appropriate template                   
                    template = "D:\Uni\PhD\models\Toolboxes\Templates.gdb\L{0}_HIP".format(len(np_array.shape))
                    #rotate and scale
                    arcpy.AddMessage(tuple(array.title))
                    e_shift, n_shift  = self.__HIP2xy__(tuple (int(val) for val in array.title))
                    scale = self.cellsize*7**(0.5*L)
                    rotation = degrees(atan(3**.5/2))*L
                    #need to Identify the latitude and longitude of the origin in the projection of the template projection. Can't remember why?
                    long_centre, lat_centre = 0, 0
                    shift_projection = "PROJCS['Rotation_Projection',GEOGCS['GCS_WGS_1984',DATUM['D_WGS_1984',SPHEROID['WGS_1984',6378137.0,298.257223563]],PRIMEM['Greenwich',0.0],UNIT['Degree',0.0174532925199433]],PROJECTION['Local'],PARAMETER['False_Easting',{0}],PARAMETER['False_Northing',{1}],PARAMETER['Scale_Factor',{2}],PARAMETER['Azimuth',{3}],PARAMETER['Longitude_Of_Center',{4}],PARAMETER['Latitude_Of_Center',{5}],UNIT['Meter',1.0]]".format(e_shift, n_shift, scale, rotation, long_centre, lat_centre)
                    
                    #translate by project and redefine
                    arcpy.Project_management(template, fc_fullname, shift_projection) 
                    arcpy.DefineProjection_management(fc_fullname, spatial_ref)
                    
                    #Apply value and Field calc the representation
                    field_list = arcpy.ListFields(fc_fullname) #list of field objects with .name attributes
                    for field in field_list:
                        print field.name                          

                    short_list = filter(lambda x: "hip" in x, [i.name for i in field_list])#filter out those with hip_ in, may give false positives if the name of the file has hip_ in it
                    short_list.sort(reverse = True)#put them in order
                    
                    rows = arcpy.da.UpdateCursor (fc_fullname, short_list+['u','value', 'RuleID'])
                    print len(short_list)
                    for row in rows: 
                        HIP = tuple(row[:-3]) #evaluate the key
                        row[-2]  = array[HIP]
                        row[-3] = 4
                        print row[-2]
                        if row[-2] == self.NoDataValue:
                            row[-1] = 33
                            print 'NoData'
                        else:
                            print 'false'
                            try:
                                row[-1] = max(1,int((row[-2]-minimum)*32/(maximum-minimum))) #32 is number of rules in the representation
                                print row[-1]
                                print row
                            except:
                                arcpy.AddMessage("Value Rule problem: {0}, Min: {1}, Max: {2}".format(row[-2], minimum, maximum))
                        rows.updateRow(row)
                    
                    # Delete cursor and row objects to remove locks on the data 
                    del row, rows
             
            #Move the files into the feature dataset and delete the temporary copies. 
            #Featurecalss to featureclass seemed to work before but now needs the field mapping to be set explicitly because the representation field is not being mapped. One possible solution is to use a fieldmappings object. (see FieldMappings (arcpy))

            for part in parts:
                 
                try:
                    arcpy.Copy_management (part, '{0}\\{1}'.format(self.fname, os.path.basename(part)[:-1]))#slice out the tempfile trailing underscore
                #By using the name as spatial reference quicker copy can be used instead of the below line.
                except:
                    arcpy.FeatureClassToFeatureClass_conversion (part, self.fname, os.path.basename(part)[:-1])#alternative / safer copy, will handle some refernece conficts
                    arcpy.AddWarning(os.path.basename(part)[:-1] + " has an anomolous spatial reference")
#                try:
#                    arcpy.Delete_management(part) 
#                except:
#                    arcpy.AddWarning('Unable to delete '+part)
                            
    def drawDissolvedLevels():
        """This tool creates pyramid featureclasses. However instead of drawing them, it merges the bass layer to create fractal shapes
        Be wary of large datasets creating very large polygons with many sides.
        Requires the base level to be drawn
        Requires Pyramid Layers to be calced
        Lazy way to use the dissolve stats to get value: limited options + may not match pyramids
        Not complete
        """
        #check if base layer is drawn
        #from start to stop
            #do the dissolve
            #Get values from Pyramid
        #
#    def HeptTree(self, value=0):
#        #calculates the heptree of the dataset based on the rule and value provided, for instance mean within value.
#        #Output is a featureclass need to be adaptive if there are many polygons in output
#        #assumes pyimids exist
#        #Not working -depreciated use sparse2
#        
#        #select the pyrimid fcs
#        #If does not exist creates one
#        arcpy.env.overwriteOutput = True
#        gdb, fname = os.path.split(self.fname)
#        spatial_ref = arcpy.SpatialReference()
#        spatial_ref.loadFromString(self.projection)
#        
#        if arcpy.Exists(self.fname):
#            arcpy.env.workspace = self.fname            
#        else:
#            arcpy.CreateFeatureDataset_management (gdb, fname, spatial_ref)
#            arcpy.env.workspace = self.fname
#            
#        #create fetureclass for result, ends with _var for variable resolution
#        #will need different template if no fd
#
#        template = arcpy.ListFeatureClasses("*{0}".format("0"*(self.level+1)))
#        arcpy.AddMessage('Used as template: '+ template[0])
#        if len(template) ==0:
#            arcpy.AddError('{0} contains no featureclasses ending in _{1}'.format(self.fname, "0"*(self.level+1)))#need to fix, should not need +1 here?
#        else:
#            arcpy.CreateFeatureclass_management(self.fname, fname+'_var', "POLYGON", template[0],"DISABLED","DISABLED",spatial_ref)
#            
#        where = (0,)
#        with tables.openFile(self.h5, mode = 'r') as op:
#            open_HHSM = openHSM(op, self)
#            open_HHSM.__cascade__(where, value)


        
class RHSM(HSM):
    """Extends HSM for Rectagonal specific features"""
    
    #Class variable
    #These variables do not change they are stored here to prevent contant redefinition
    aperture = 9 #tree bredth
    
    #hip to xy
    rotations = {0:numpy.array([[0, 0], [0, 0]]), 1: numpy.array([[1, 0], [0, 1]]), 2: numpy.array([[1, -1], [1, 1]]), 3: numpy.array([[0, -1], [1, 0]]), 4: numpy.array([[-1, -1], [1, -1]]), 5: numpy.array([[-1, 0], [0, -1]]), 6: numpy.array([[-1, 1], [-1, -1]]), 7: numpy.array([[0, 1], [-1, 0]]), 8: numpy.array([[1, +1], [-1, +1]])}
    c = numpy.array([[1],[0]])
    
    #xy corners
    cnr = numpy.array([[0, .5, -.5, -.5, .5], [0, .5, .5, -.5, -.5]])
    
    #hip add/subtract
    IPadd_table = numpy.array([[(0,0),(1,0),(2,0),(3,0),(4,0),(5,0),(6,0),(7,0),(8,0)],
                                [(1,0),(5,1),(4,1),(2,0),(3,0),(0,0),(7,0),(8,0),(6,1)],
                                 [(2,0),(4,1),(6,2),(8,3),(7,3),(3,0),(0,0),(1,0),(5,1)],
                                  [(3,0),(2,0),(8,3),(7,3),(6,3),(4,0),(5,0),(0,0),(1,0)],
                                   [(4,0),(3,0),(7,3),(6,3),(8,4),(2,5),(1,5),(5,0),(0,0)],
                                    [(5,0),(0,0),(3,0),(4,0),(2,5),(1,5),(8,5),(6,0),(7,0)],
                                     [(6,0),(7,0),(0,0),(5,0),(1,5),(8,5),(2,6),(4,7),(3,7)],
                                      [(7,0),(8,0),(1,0),(0,0),(5,0),(6,0),(4,7),(3,7),(2,7)],
                                       [(8,0),(6,1),(5,1),(1,0),(0,0),(7,0),(3,7),(2,7),(4,8)]])
    IPnegation_table = [0, 5, 6, 7, 8, 1, 2, 3, 4]
    
    #Nieghbourhood operations
    nieghbour_dist = numpy.array([1, 1, 2**0.5, 1, 2**0.5, 1, 2**0.5, 1, 2**0.5])#First value is really 0 but this would create 0 div error for drop calculations
    
    #Dinf
    e0 = [0, 0, 0, 0, 0, 0, 0, 0]
    e1 = [1, 3, 3, 5, 5, 7, 7, 1]
    e2 = [2, 2, 4, 4, 6, 6, 8, 8]
    ac = [0, 1, 1, 2, 2, 3, 3, 4]
    af = [1,-1, 1,-1, 1,-1, 1,-1]
    

    def valuesFrom_XY(self, XY_array, X0, Y0):
        """Takes the values from the XYarray and populates the h5 file
        XY_array is a numpy array with XY indexing.
        X0, Y0 are the col, row of the pixel that is the origin of in UV
        No allowance for cell size"""
        
        #Iterate the HSM
        with tables.openFile(self.h5, mode = "r+") as h5file:
            #create a conversion object
            for array in h5file.walkNodes("/L0", "Array"):
                np = array.read()
                it =  numpy.nditer(np, flags=['multi_index'], op_flags=['readwrite'])
                while not it.finished:
                    HIP = tuple (int(val) for val in array.title[:-self.agg])+it.multi_index
                    (x, y) = self._convIP2xy(HIP)
                    if not 0 <= x+X0 < XY_array.shape[0] or not 0 <= y+Y0 < XY_array.shape[1]:
                        it[0] = self.NoDataValue
                    else:
                        xcell, ycell  = int(x+X0), int(-y+Y0)
                        it[0]  = XY_array[ycell, xcell]
                    it.iternext()                
                array[:]=np
                array.flush()
                
    def hexCondat(self, out_fname, out_level, out_agg):
        """#Produces a new HHSM with filaename, level and agg provided
        #Values are extracted from the RHSM and converted by a condat H22 fractional delay filter
        #If output arrayis larger than the input array values will be assigned to nodata outside the data area
        Does not support arrays larger than RAM"""
        #Create array for rect values
        print 'converting to hex'
        new_shape = ((9**self.level)**0.5, (9**self.level)**0.5) #how large?
        data = numpy.zeros(new_shape)
        origin = ((data.shape[0]-1)/2,)*2
        pad_val = origin[0]
        #transfer value
        with tables.openFile(self.h5, mode = "r") as h5file:
            op = openHSM(h5file, self)
            for i in op:
                xy = self._convIP2xy(i[1])
                #convert to int so can be used as index
                c  = int(xy[0] + origin[0])
                r  = int(-xy[1] + origin[1])
                #print r.dtype, c
                data[r,c] = i[0]
            del op
        pad_val = int(pad_val) #or some reason np.pad does not like type Long   
        datapad = numpy.pad(data, pad_val, mode = 'reflect')         
        #Condat
        a = 3**0.5-(6/3**0.5)**0.5
        b = (3**0.5/6.0)**0.5
        c = 2-(6/3**0.5)**0.5 
        origin = (origin[0]+pad_val, origin[1]+pad_val)
        hex_data = condat.condat(datapad, a, b, c, origin)
        #Create HHSM
        out_HSM = HHSM (out_fname)
        out_HSM.level = out_level
        out_HSM.agg = out_agg 
        out_HSM.projection = self.projection
        out_HSM.x0 = self.x0
        out_HSM.y0 = self.y0
        out_HSM.NoDataValue = self.NoDataValue
        out_HSM.cellsize = self.cellsize*(2/3**0.5)**0.5
        out_HSM.CreateHIP()
        #transfer values
        (lx, ux) = (origin[0] - pad_val - new_shape[0], new_shape[0] - origin[0] + pad_val) 
        (ly, uy) = (origin[1] - pad_val - new_shape[1], new_shape[1] - origin[1] + pad_val)
        limit = "{0} <= x <= {1} and {2} <= y <= {3}".format(lx+1, ux-1, ly+1, uy-1)
        #limit = None
        out_HSM.valuesFrom_UVacute(hex_data, origin[0], origin[1], limit)
        
        return out_HSM
        
                
                
    def _IP2xy(self, result, i, val):
        result = numpy.add(result, numpy.dot(self.rotations[val]*3**i,self.c))
        return result
        
    def _calcCorners(self, centre_ip, pyr, origin):
        centre = numpy.array([[0],[0]])
        for i, val in enumerate(centre_ip[::-1]):
            centre = self._IP2xy(centre, i, val)
#        u, v = result
#        centre = numpy.array([[u],[v]])
        corners = numpy.add(centre, self.cnr* 3**pyr)
        return numpy.add(self.cellsize*corners, origin).transpose()
        
#    def dinf(self, hood):
#        #calculates the dinf direction of the hood in vector equivalents (ie ranges from 1 to aperture-1)
#        #calculate the directions and slopes
#        faces = numpy.array([[0.0, 0.0]]*(self.aperture-1))
#        for i in range (self.aperture-1):
#            s1i = self.s1(hood[self.e0[i]], hood[self.e1[i]], self.cellsize)
#            s2i = self.s2(hood[self.e1[i]], hood[self.e2[i]], self.cellsize)
#            ri = atan2(s2i, s1i)
#            si = (s1i**2 + s2i**2)**0.5
#            if ri < 0:
#                ri = 0
#                si = s1i
#            elif ri > atan(1):
#                ri = atan(1)
#                si = (hood[self.e0[i]] - hood[self.e2[i]])/float((2*self.cellsize**2)**0.5)
#            faces[i] = [ri, si]
#        dir_index = list(faces[:,1]).index(max(faces[:,1]))#this line identifies (first) index of the steepest slope
#        if faces[dir_index][1] < 0:
#            direction = 0
#        else:
#            dir_inf = self.af[dir_index]*faces[dir_index][0] + self.ac[dir_index]*pi/2.0
#            direction = dir_inf/atan(1) + 1 #convert to HSM
#        return direction
#        
#            
#        
#    def s1(self, e0i, e1i, d1i):
#        #used to determine the slope of side one of the dinf facet
#        result = (e0i - e1i)/float(d1i)
#        return result
#        
#    def s2(self, e1i, e2i, d2i):
#        #used to determine the slope of one side two of the dinf facet
#        result = (e1i - e2i)/float(d2i)
#        return result
        
class THSM(HSM):
    """Extends HSM for triangle specific features"""
    #Class variable
    #These variables do not change they are stored here to prevent constant redefinition
    aperture = 4 #tree bredth
    
    #hip to xy
    rotations = {0:numpy.array([[0, 0], [0, 0]]), 1: numpy.array([[0, -1], [1, 0]]), 2: numpy.array([[-3**0.5/2.0, 0.5], [-0.5, -3**0.5/2.0]]), 3: numpy.array([[3**0.5/2.0, 0.5], [-0.5, 3**0.5/2.0]])}
    b = numpy.array([[2, 0],[0, -2]])
    #b = numpy.array([[0, 1],[-1, 0]])
    c = numpy.array([[1],[0]])
    
    #xy corners
    cnr = numpy.array([[0, 0, 3**0.5/2.0, -3**0.5/2.0], [0, 1, -.5, -.5]])
    
    #hip add/subtract
    IPadd_table = numpy.array([[(0,0),(1,0),(2,0),(3,0)],
                                [(1,0),(0,0),(3,2),(2,3)],
                                 [(2,0),(3,1),(0,0),(1,3)],
                                  [(3,0),(2,1),(1,2),(0,0)]])
    IPnegation_table = [0, 1, 2, 3]
    
    #Nieghbourhood operations
    nieghbour_dist = numpy.array([1, 1, 1, 1])#First value is really 0 but this would create 0 div error for drop calculations
    
    def _IP2xy(self, result, i, val, flip=0):#The flip value is true if the IP value is not in IP = 0 in IP+1 Place value. If true the base vector is "flipped" to point down.
        if flip == -1:
            result = numpy.add(result, numpy.dot(numpy.dot(self.rotations[val], numpy.linalg.matrix_power(self.b, i)), self.c))
        elif flip == 1:
            result = numpy.add(result, numpy.dot(numpy.dot(self.rotations[val], numpy.linalg.matrix_power(self.b, i)), -self.c))
        else:
            print 'no flip value'
        return result
        
    def _convIP2xy(self, ip):      
        result = numpy.array([[0],[0]])
        for i, val in enumerate(ip[::-1]):
            result = self._IP2xy(result, i, val, flip = (-1)**(len(ip) - ip[-i-2::-1].count(0)))
        return result
        
    def _calcCorners(self, centre_ip, pyr, origin):
        
        centre = numpy.array([[0],[0]])
        for i, val in enumerate(centre_ip[::-1]):
            centre = self._IP2xy(centre, i, val, flip = (-1)**(len(centre_ip) - centre_ip[-i-2::-1].count(0)))
        #centre triangle is pointing the other way if there is an odd number of non 0s preceding it 
        cnr_flip = -((-1)**(len(centre_ip) - centre_ip.count(0)))
        corners = numpy.add(centre, numpy.dot (numpy.linalg.matrix_power (self.b, pyr), self.cnr*cnr_flip))#This is lazy and flips the horizontal axis too but no effective difference
        return numpy.add(self.cellsize*corners, origin).transpose()
        
    def Dinf2radians(self, output, start_level = 0, stop_level = 1):
        #Converts calculated RHSM-tri dinf values into their equivalent radian direction (clockwise from x-axis)
        
        print "Canging to radians"     
        if not stop_level:
            stop_level = self.HSM_object.level        
        
        #create the output HHSM
        out = copy.deepcopy(self)
        out.h5 = output+'.h5'
        out.fname = output
        #delete h5 file it currently has the copied values.
        if os.path.exists(out.h5):
            os.remove(out.h5)
        out.CreateHIP()
        if stop_level > 1:
            out.PyrTree()#Creates blank files
            
        #open the input direction data
        with tables.openFile(self.h5, mode = "r") as in_file:
            #open the output data
            with tables.openFile(out.h5, mode = "r+") as out_file:
                    #go through the levels
                for level in range(start_level, stop_level):
                    base = "/L{0}".format(level)
                    #walk through the arrays on the specified level           
                    for in_array in in_file.walkNodes(base, "Array"):
                        #create an input numpy array
                        in_np = in_array.read()
                        #find its path name
                        out_array = out_file.getNode(in_array._v_pathname)
                        #get the output array with the same name
                        out_np = out_array.read()
                        #iterate the input elevation data
                        it =  numpy.nditer(in_np, flags=['multi_index'], op_flags=['readwrite'])
                        while not it.finished:
                            #find its IPord
                            cell = self._HHSM2HIP(in_array._v_pathname[len(str(level))+2:], in_array.name, it.multi_index)
                            flip = -((-1)**(len(cell) - cell.count(0)))
                            if it[0] == self.NoDataValue:
                                out_np[it.multi_index] = self.NoDataValue
                            else:
                                
                                if flip == 1:
                                    out_np[it.multi_index] = (in_np[it.multi_index]+pi)%(2*pi)
                                else:
                                    out_np[it.multi_index] = (in_np[it.multi_index])
                            it.iternext()
                            
                        #Update the output array (may be unnecassary)
                        out_array[:]=out_np
                        out_array.flush()                                               
        return out


        
class openHSM(object):
    '''Operations that require an open h5 file, Need to open and close outside.
    #This is particularly used in situations where calls need to be made repeatedly to open the HSM
    #possible future edit to make a routine in the main class that creates this. Even open by calling if possible with form with open ** as.'''
    
   
    def __init__(self, open_h5, HSM):
        self.open_h5 = open_h5
        self.HSM_object = HSM
        self.IP_object = calcHSM(HSM)

        #ram shuffler, I don't think I am using this.
        #ram_lim = 10 #if 32 bit then the ram_lim is 8 as there is no overhead available beyond that, otherwise depends on ram available. 
        #self.current_dict = {}

        #self.current = [""]*HSM.aperture**(min(ram_lim, self.HSM_object.level)-min(ram_lim, self.HSM_object.agg))
        
        #An extra shuffler for functions that need to use both L0 and pyr arrays
        #self.current_dict_1 = {}
        #self.current_1 = [""]*HSM.aperture**(min(ram_lim, self.HSM_object.level)-min(ram_lim, self.HSM_object.agg))        
        
        
    def __iter__(self, flags=['IP'], level = 0, op_flags=['readwrite']):
        #Iterator, not really read write.
        #Iterates the supplied level
        if flags==['HSM']:
            #In adition to the value the HSM position is returned as triplet
            for array in self.open_h5.walkNodes("/L{0}".format(level), "Array"):
                np = array.read()
                it =  numpy.nditer(np, flags=['multi_index'], op_flags=op_flags)
                while not it.finished:
                    yield [it[0], array.parentnode, array.name, it.multi_index]
                    it.iternext()
                    
        if flags==['IP']:
            #In adition to the value the HHSM position is returned as tuple
            for array in self.open_h5.walkNodes("/L{0}".format(level), "Array"):
                np = array.read()
                it =  numpy.nditer(np, flags=['multi_index'], op_flags=op_flags)
                while not it.finished:
                    yield [it[0], tuple (int(val) for val in array.title[:-self.HSM_object.agg])+it.multi_index]
                    it.iternext()

           
    def _Neighbourhood(self, cell, level=0, stat = 'val'):
        #creates a list of values for the nieghbourhood, order ie index is equal to the direction of nieghbour  first item is the centre
        hood = [self.HSM_object.NoDataValue]*self.HSM_object.aperture
        base = "/L{0}".format(level)
        
        #basic val test: returns the nieghbouring values
        if stat == 'val':
            for i in range(self.HSM_object.aperture):
                #print cell, (i,)
                hood_IP = self.IP_object.IPadd(cell, (i,), self.HSM_object.level-level) 
                where, name, index = self.HSM_object._HIP2HHSM(hood_IP)
                table = self.open_h5.getNode(base+where, name)
                #print index, table[index], self.HSM_object.level
                try:
                    hood[i]=table[index]
                except:
                    pass
        elif stat == 'drop':
        #Returns the slope of descent in array units, useful for D8 type solutions
            hood = numpy.array(hood, dtype = 'float64')
            for i in range(self.HSM_object.aperture): 
                hood_IP = self.IP_object.IPadd(cell, (i,), self.HSM_object.level-level)
                where, name, index = self.HSM_object._HIP2HHSM(hood_IP)
                table = self.open_h5.getNode(base+where, name)
                try:
                    hood[i]=table[index]
                except:
                    #edge error
                    pass
            if not numpy.equal(hood, self.HSM_object.NoDataValue).any():  
                hood = (hood-hood[0])/self.HSM_object.nieghbour_dist
            
            hood = list(hood)
                
        return hood
        
    def _VarNeighbourhood(self, cell, stat = 'val'):
        """
        Produces neighbourhoods as a list of (val, dir, delta level) tuples
        Extracts values from the sparse array (variable density).

        Parameters
        ----------
        cell : tuple
        (int, int, int...)
        IP index of the cell for which the neighbourhood will be defined
        stat : string from list ['val, 'drop']
        if 'val then produces the value of the nieghbour 
        if 'drop' produces the signed difference between the cell and the neighbour
        
        Returns
        -------
        hood : List
        [val, dlevel, dir]
        val : numerical
        the value of the nieghbour
        dlevel : Int
        The difference in level between neighbour and cell (0 if same)
        dir : tuple
        (int, int...)
        The direcion in which the neighbour lies. 
        If dlevel >= 0, (int,) representing the IP direction on the appropriate level
        If dlevel < 0, ie. nieghbour is at a finer resolution (int, int...) starting with the coarsest step 
           
        Notes
        -----
        """
        
        hood = []      
        #basic val test: returns the nieghbouring values
        if stat == 'val':
            for i in range(1, self.HSM_object.aperture):
                external = False
                hood_IP = self.IP_object.IPadd(cell, (i,), len(cell))
                curr_node = self.open_h5.getNode("/", "S")
                for ordinate in hood_IP:
                    if str(ordinate) in curr_node:
                        curr_node = self.open_h5.getNode(curr_node, str(ordinate))
                        if type(curr_node) == tables.array.Array:
                            IP = tuple(int(x) for x in curr_node._v_pathname.split('/')[2:])
                            if not IP in (x[0] for x in hood): 
                                hood.append([IP, curr_node.read()[0]])
                            break
                    else:
                        #Nieghbour is external to the study area
                        external = True                       
                        break

                IP = tuple(int(x) for x in curr_node._v_pathname.split('/')[2:])

                if not type(curr_node) == tables.array.Array and external == False:
                    #In this case some nieghbours are of finer resolution
                    self._VhoodCall(cell, IP, curr_node, hood)
        return hood

                             
    def _VhoodCall(self, cell, IP, curr_node, hood_list):
        #Recursive tool for variable neighbourhood determining when neighbour is finer
        #Makes use of aliasing, may not be a good idea
        
        #for each child (can ignore centre)
        for i in range(1, self.HSM_object.aperture):
            child_IP = IP + (i,)
            #determine the neighbours           
            for j in range(1, self.HSM_object.aperture):               
                hood_IP = self.IP_object.IPadd(child_IP, (j,), len(child_IP))
                #Does the address begin with cell address and not in hood
                if hood_IP[:len(cell)] == cell and not child_IP in (x[0] for x in hood_list):
                    if str(i) in curr_node:#otherwise not in sparse
                        curr_node1 = self.open_h5.getNode(curr_node, str(i))
                        if not type(curr_node1) == tables.array.Array:
                            self._VhoodCall(cell, child_IP, curr_node1, hood_list)
                        else:
                            hood_list.append([child_IP, curr_node1.read()[0]])
                        #and no children
                        #if so add to hood
                        #else repeat
                                
        
#    def _cascade(self, where, limit):
#    #recursive traverse for HeptTree
#    #start at top iterate through cells and cascade through levels
#    #Not working or currently being used
#        for i in range (7):
#            #If criteria is met, draw the voronoi in output and move on
#            
#            #First determine the HHSM address of the pyr value and the main array
#            where = where + (i,)
#            main_where1, main_name, main_index = self.HHSM_object.__HIP2HHSM__(where)
#            print '{0}, {1}, {2}, {3}'.format(main_where1, main_name, main_index, where)
#            main_where = '/L0'+main_where1
#            pyr_level = self.HHSM_object.level-len(where)
#            pyr_agg = min(self.HHSM_object.agg, pyr_level) 
#            print pyr_level, pyr_agg
#            pyr_where1, pyr_name, pyr_index = self.HHSM_object.__HIP2HHSM__(where, pyr_level, pyr_agg)
#            pyr_where = '/L{0}{1}'.format(pyr_level,pyr_where1)
#            print '{0}, {1}, {2}, {3}'.format(pyr_where1, pyr_name, pyr_index, where)
#            #RAM shuffle main
#            if main_where not in self.current:                
#                main_table = self.open_h5.getNode(main_where, main_name)
#                self.current_dict[main_where] = main_table.read()
#                if self.current[0]:
#                    del self.current_dict[self.current[0]]
#                self.current = self.current[1:]+[main_where]
#            #RAM shuffle pyr    
#            if pyr_where not in self.current_1:                
#                pyr_table = self.open_h5.getNode(pyr_where, pyr_name)
#                self.current_dict_1[pyr_where] = pyr_table.read()
#                if self.current_1[0]:
#                    del self.current_dict_1[self.current_1[0]]
#                self.current_1 = self.current_1[1:]+[pyr_where]
#
#            #Condition check
#            print self.current, self.current_1
#            print main_where, main_index, 
#            print pyr_where, pyr_index
#            #print self.current_dict[main_where][main_index]
#            print self.current_dict_1[pyr_where][pyr_index]
#            print limit
#            if numpy.allclose(self.current_dict[main_where][main_index], self.current_dict_1[pyr_where][pyr_index], atol = limit):
#                #draw feature and continue traverse
#                print 'drawing'
#                centre = self.HHSM_object.__HHSM2HIP__(main_where, main_name, main_index)
#                print centre, pyr_level
#                print
#            else:
#                #cascade
#                self.__cascade__(where+(i,), limit)
                
    def _get_vals(self, zone, level = 0, val_type = 'L'):
        """returns a numpy array of the values in a given level,zone
        If the zone is shorter than the level more than one value will be returned
        If val_type == 'S' level is irrelavent
        Cannot get vals across multiple arrays in one go
        """
        
        agg = min(self.HSM_object.agg, level)
        if val_type in ['L', 'E']:
            where, name, index = self.HSM_object._HIP2HHSM(zone, level, agg) 
        elif val_type == 'S':
            level = ""
            where, name, index = ''.join('/'+str(x) for x in zone[:-1]), str(zone[-1]), 0
        array = self.open_h5.getNode('/{0}{1}{2}'.format(val_type, level, where), name)
        out_array = array[index]
        if out_array.shape == ():#Avoid 0-shape arrays
            out_array = out_array.reshape(1,) 

        return out_array
    
    
    def _set_vals(self, vals, zone, level = 0, val_type = 'L'):
        #places the vals into the array
        #level is the pyramid level, float or int
        #zone is a hip ordinate as a tuple
        #Vals is a value or array
        #vals may be broadcast
        
        agg = min(self.HSM_object.agg, self.HSM_object.level-level)
        if val_type in ['L', 'E']:
            where, name, index = self.HSM_object._HIP2HHSM(zone, self.HSM_object.level-level, agg)
        elif val_type == 'S':
            level = ""
            where, name, index = ''.join('/'+str(x) for x in zone[:-1]), str(zone[-1]), 0            
        array = self.open_h5.getNode('/{0}{1}{2}'.format(val_type, level, where), name)
        array[index] = vals
        array.flush()
        
        
    def _toleranceCheck(self, zone, level, tolerance):
        #recursive tolerance check writes to sparse and stops search if tolerance is met.
        
        if level > 0:
            vals = self._get_vals(zone, level, val_type = 'E')
            for i, val in enumerate(vals):
                if val > tolerance or val == self.HSM_object.NoDataValue:
                    self._toleranceCheck(zone+(i,), level-1, tolerance)
                else:#write the value to the node
                    out_val = self._get_vals(zone+(i,), level, val_type = 'L')
                    array_name = str(i)
                    array_where ="/S"+''.join('/'+ str(x) for x in zone)
                    self.open_h5.createArray(array_where, array_name, out_val, createparents=True)
        else:
            out_vals = self._get_vals(zone, level, val_type = 'L')
            for i, out_val in enumerate (out_vals):
                if not out_val == self.HSM_object.NoDataValue:
                    array_name = str(i)
                    array_where ="/S"+''.join('/'+ str(x) for x in zone)
                    self.open_h5.createArray(array_where, array_name, numpy.array([out_val]), createparents=True)
                                    
        
        
class calcHSM:
    #object for handling various hip calculations
    #Used to prevent repeatedly recreating identical numpy arrays
    
    def __init__(self, HSM_object):
        #general
        self.origin = numpy.array([[HSM_object.x0], [HSM_object.y0]])
        self.HSM_object = HSM_object
        
        #xy rotation
        self.rot = numpy.array([[cos(HSM_object.rotation), -sin(HSM_object.rotation)],[sin(HSM_object.rotation), cos(HSM_object.rotation)]])
                
        #Used to record the arrays in main memory, limits time spent converting pytable nodes to numpy arrays (where necessary 
        ###I don't think i use this###
        #self.current_dict = {}
        #self.current = [""]*7**(min(10,HSM_object.level)-min(10, HSM_object.agg))#32 bit if the level is greater than 8 there is no overhead available, the 10 depends on ram availabe.
    
    def IPadd(self, a, b, min_length = 0):
        #performs HIP addition
        #input: hip tuple addends (x2), min length of output
         #use min length to mantain leading zeros if required
        #Output: hip tuple  (a +b) of min length
       
            
        #reverse the inputs to make indexing easier
        a1=a[::-1]
        b1=b[::-1]
        
        length = max(len(a1),len(b1))
        i=0
        result = [0]*length#initialise result
        #intialise place value
        columns=[[] for col in range(length)] 
            
        #put the inputs into the columns
        for item , value in enumerate(a1):
            columns[item] += [value]
        for item , value in enumerate(b1):
            columns[item] += [value]
    
        #The addition sequentially sums each column with the corresponding result value
        while i < length:
    
            if any(columns[i]):#do the sum if there are non zero values in col
                # add a column for place value overflow
                columns+=[[]]
                length += 1
                result += [0]
                
                try:
                    for addend in columns[i]:#for each value in the column
                        columns[i+1] += [self.HSM_object.IPadd_table[result[i],addend][1]]  
                        result[i] = self.HSM_object.IPadd_table[result[i],addend][0]
                except IndexError:
                    raise
                    print i, self.HSM_object.IPadd_table[result[i]], addend
            i+=1
        reverse = result[::-1]
        #strip the leading zeros
        while reverse[0] == 0 and len(reverse) > max(len(a1), len(b1), min_length): #retain the level of longest input
            del(reverse[0])
        
        return tuple (reverse)
               
    def IPadd2():
        #there must be a quicker way
        pass

    def IPnegation(self, hip):
        #determines the negation ie the vector with the opposite sense
        #useful for subtraction (just add the negation)
        #input HIP tuple
        #output HIP tuple
        out = ()
        for l in hip:
            out += (self.HSM_object.IPnegation_table[l],)           
        return out        
               
    def xy_corners(self, centre_ip, pyr = 0):
        #input an IP index as tuple
        #output a list of tuples, each tuple being the cartesian coordinates of a corner. The order of the list is [centre, first corner anticlockwise (looking down) (-should change to clockwise) from base vector for level. Length depends on IP type
        #Goes through uvAcute. Not yet sure if corners can be defined in HIP

        corners = self.HSM_object._calcCorners(centre_ip, pyr, self.origin)
        return corners

    def IP2xy(self, ip):
        #input a HIP index as tuple
        #output xy coords as tuple
        
        #convert to IP   
        result = self.HSM_object._convIP2xy(ip)
        #apply rotation    
        if self.HSM_object.rotation:
            result = numpy.dot(self.rot, result)
        #apply origin translattion
        result = numpy.add(self.HSM_object.cellsize*result, self.origin)
            
        return(tuple(result.flatten()))

#    def IP2xy(self, ip):
#        #input a HIP index as tuple
#        #output xy coords as tuple
#        result = numpy.array([[0],[0]])
#        for i, val in enumerate(ip[::-1]):
#            result = self.HSM_object._IP2xy(result, i, val)
#        #apply rotation    
#        if self.HSM_object.rotation:
#            result = numpy.dot(self.rot, result)
#            
#        return(tuple(numpy.add(self.HSM_object.cellsize*result, self.origin).flatten()))


    def xy2IP(self, x, y):
        #Closest to technique for each of the levels of the hip
        #Ideally, should be replaced by an analytical technique
        #breifer form for RHSM possible
        ip_ord = [0]*self.HSM_object.level      
        for i in range(self.HSM_object.level):
            #determine the centres on this level and pick the closest
            candidates = []
            for j in range(7):
                ip_ord[i] = j
                xy_coord = self.ip2xy(ip_ord)
                candidates += [((xy_coord[0]-x)**2 + (xy_coord[1]-y)**2)**0.5]
            candidate = candidates.index(min(candidates))
            #Need to check for for disolve/voronoi discrepencies 4 levels above
            ip_ord[i] = candidate
            if i > 2:
                #check its neighbours on that level
                hood = {}
                ip_ord = tuple(ip_ord)
                hood[ip_ord] = min(candidates)
                for j in range(1,7):
                    add_end = (0,)*i+(j,) + (0,)*(self.HSM_object.level-1-i)
                    neighbour = self.IPadd(add_end, ip_ord)
                    xy_coord = self.ip2xy(neighbour)
                    hood[neighbour] = ((xy_coord[0]-x)**2 + (xy_coord[1]-y)**2)**0.5
                ip_ord = min(hood, key=hood.get) 
                ip_ord = list(ip_ord)
                
        return tuple(ip_ord)
        
        
class hydro:
    #performs core hydrological functions
    
    def __init__(self, HSM_object):
        self.HSM_object = HSM_object
                
    def FlowDir(self, output, start_level = 0, stop_level = 1):
    #determines D8 flow dir
    #Will overwrite output
    #Will calculate lods seperately based on the elevation pyrimid if stop level is set.
    #stop level is exclusive and determines how many levels to solve.   
    #Not working if agg < level!   
        print "Determining Dx flow direction"     
        if not stop_level:
            stop_level = self.HSM_object.level        
        
        #create the output HHSM
        out = copy.deepcopy(self.HSM_object)
        out.h5 = output+'.h5'
        out.fname = output
        #delete h5 file it currently has the copied values.
        if os.path.exists(out.h5):
            os.remove(out.h5)
        out.CreateHIP()
        if stop_level > 1:
            out.PyrTree()#Creates blank files
            
        #open the input elevation data
        with tables.openFile(self.HSM_object.h5, mode = "r") as in_file:
            #open the output flow direction data
            with tables.openFile(out.h5, mode = "r+") as out_file:
                #create an openHSM object with the input elevation data
                op = openHSM(in_file, self.HSM_object)
                
                    #go through the levels
                for level in range(start_level, stop_level):
                    base = "/L{0}".format(level)
                    #walk through the arrays on the specified level           
                    for in_array in in_file.walkNodes(base, "Array"):
                        #create an input numpy array
                        in_np = in_array.read()
                        #find its path name
                        out_array = out_file.getNode(in_array._v_pathname)
                        #get the output array with the same name
                        out_np = out_array.read()
                        #iterate the input elevation data
                        it =  numpy.nditer(in_np, flags=['multi_index'], op_flags=['readwrite'])
                        while not it.finished:
                            #find its IPord
                            cell = self.HSM_object._HHSM2HIP(in_array._v_pathname[len(str(level))+2:], in_array.name, it.multi_index)
                            #determine the drop neighbourhood 
                            hood = op._Neighbourhood(cell[-min(self.HSM_object.level, self.HSM_object.level-level+1):], level, stat = 'drop')#Very ugly way to remove leading zero from path
                            #assign the index of the minimum value as the output direction
                            if numpy.equal(hood, self.HSM_object.NoDataValue).any():
                                out_np[it.multi_index] = self.HSM_object.NoDataValue
                            else:
                                out_np[it.multi_index] = hood.index(min(hood))
                            it.iternext()
                            
                        #Update the output array (may be unnecassary)
                        out_array[:]=out_np
                        out_array.flush()                                               
        return out
        
    def Dx2Dinf(self, inDxFname, output, start_level = 0, stop_level = 1):
        """
            Converts Dx values into radian directions (clockwise-normalised to 2pi radians)
            RHSM-tri is strange
        """
        arcpy.AddMessage('Converting Dx values of {0} to {1}'.format(inDxFname, output))
        
        out = copy.deepcopy(self.HSM_object)
        out.h5 = output+'.h5'
        out.fname = output
        #delete h5 file it currently has the copied values.
        if os.path.exists(out.h5):
            os.remove(out.h5)
        out.CreateHIP()
        if stop_level > 1:
            out.PyrTree()#Creates blank files
            
        #open the input data
        with tables.openFile(self.HSM_object.h5, mode = "r") as in_file:
            #open the output data
            with tables.openFile(out.h5, mode = "r+") as out_file:
                #create an openHSM object with the input elevation data
                print "files open"                
                    #go through the levels
                for level in range(start_level, stop_level):
                    base = "/L{0}".format(level)
                    #walk through the arrays on the specified level           
                    for in_array in in_file.walkNodes(base, "Array"):
                        #create an input numpy array
                        in_np = in_array.read()
                        #find its path name
                        #get the output array with the same name 
                        out_array = out_file.getNode(in_array._v_pathname)                                  
                        #mask out No Data cells and pits
                        pits = in_np == -1
                        pits = pits.reshape(in_np.shape)
                        nodatas = in_np == out.NoDataValue
                        nodatas = nodatas.reshape(in_np.shape)
                        #Apply the conversion
                        if out.aperture == 4:
                            #rotates a further 90 degrees to become x-axis aligned. Doubtful for accumulating
                            out_np = ((in_np - 1) * 2*pi/(out.aperture-1) + pi/2) % (2*pi)
                        else:
                            out_np = (in_np - 1) * 2*pi/(out.aperture-1)
                        
                        out_np[pits] = 0
                        out_np[nodatas] = out.NoDataValue
                        #set to output data
                        out_array[:]=out_np
                        out_array.flush()
        return out
        
    def Dinf2HSM(self, input_dir, output, mode = 'LINEAR', start_level = 0, stop_level = 1):
        """converts Dinf values (clockwise-normalised to 2pi radians) to HSM values Linear Angular / Area ratio values
        #Dinf (or D8) must be run first
        #Will covert all levels from start (inclusive) to stop(exclusive) 
        #If there are error values in the input they will be copied with alteration: Caution: the error values will be in radians but the actual values in RHSM.
        #If there is a sparse tree it will not be copied.
        """
        arcpy.AddMessage('Converting Dinf values of {0} to {1}'.format(self.HSM_object.h5, 'HSM'))
        ti = time.time()
        
        out = copy.deepcopy(self.HSM_object)
        out.h5 = output+'.h5'
        out.fname = output
        #delete h5 file it currently has the copied values.
        if os.path.exists(out.h5):
            os.remove(out.h5)
        out.CreateHIP()
        if stop_level > 1:
            out.PyrTree()#Creates blank files               
        
            #open the input direction
        with tables.openFile(input_dir+'.h5', mode = "r+") as in_file:
            #copy error values, (if any)
            #open the output data
            with tables.openFile(out.h5, mode = "r+") as out_file:                  
            #go through the levels and walk through the arrays on the specified level
                for level in range(start_level, stop_level):
                    #copy error values, (if any)
                    if "/E{0}".format(level) in in_file:
                        new_group = out_file.getNode("/")
                        in_file.copyNode("/E{0}".format(level), new_group, recursive = True)
                    base = "/L{0}".format(level) 
                    
                    for in_array in in_file.walkNodes(base, "Array"):
                        #create an input numpy array and get the output array
                        in_np = in_array.read()
                        in_np_shape = in_np.shape
                        out_array = out_file.getNode(in_array._v_pathname) 
                        #mask out No Data cells and pits
                        pits = in_np == -1
                        pits = pits.reshape(in_np_shape)
                        nodatas = in_np == out.NoDataValue
                        nodatas = nodatas.reshape(in_np_shape)
                        
                        #Apply the conversion
                        if mode.upper() == 'ANGULAR':
                            #ratios based on angular distance
                            print "Mode ANGULAR"
                            in_np = in_np/float(2*pi/(out.aperture-1)) + 1
                            
                        elif mode.upper() == 'LINEAR':
                            #ratios based on linear distance
                            print "Mode LINEAR"
                            if out.aperture == 7:#this is for HHSM
                                facet, theta = divmod(in_np, pi/3)
                                length = numpy.sin(theta)
                                in_np = length/(length + numpy.sin(-theta + pi/3)) + facet + 1
                                del length, theta, facet
                            elif out.aperture == 9:#this is for RHSM
                                facet, theta = divmod(in_np, pi/4)#2 if inf4, 4 if inf8
                                lengths0=facet%2
                                lengths1 = lengths0 * 2.0**0.5
                                lengths1[lengths1==0] = 1
                                lengths0[lengths0==0] = 2.0**0.5
                                in_np = lengths1*out.cellsize*numpy.sin(theta) / (lengths1*out.cellsize*numpy.sin(theta) + lengths0*out.cellsize*numpy.sin(-theta + pi/4)) + facet + 1            #open the output data
                        elif mode == 'LINEAR9_4':
                            #special call for aperture-9 4neighbourhood-Unfinished
                            if out.aperture == 9:#this is for RHSM
                                facet, theta = divmod(in_np, pi/2)#2 if inf4, 4 if inf8
                                in_np = out.cellsize*numpy.sin(theta) / (out.cellsize*numpy.sin(theta) + out.cellsize*numpy.sin(-theta + pi/2)) + facet + 1
                                
                        elif mode.upper() == 'AREA':
                            print "Mode AREA"
                            #Ratios based on triangle areas
                            if out.aperture == 7:#this is for HHSM: sin*cos most interesting
                                facet, theta = divmod(in_np, pi/3)
                                #length_tan1 = numpy.tan(theta)
                                length_sin1 = numpy.sin(theta)
                                length_cos1 = numpy.cos(theta)
                                #length_tan2 = numpy.tan(-theta + pi/3)
                                length_sin2 = numpy.sin(-theta + pi/3)
                                length_cos2 = numpy.cos(-theta + pi/3)
                                in_np = (length_sin1*length_cos1)/((length_sin1*length_cos1)+ (length_sin2*length_cos2)) + facet + 1
                                del length_sin1, length_sin2, length_cos1, length_cos2, theta, facet
                                
                            elif out.aperture == 9:#this is for RHSM
                                facet, theta = divmod(in_np, pi/4)
                                lengths0=facet%2
                                lengths1 = lengths0 * 2.0**0.5
                                lengths1[lengths1==0] = 1
                                lengths0[lengths0==0] = 2.0**0.5
                                
                                #length0_sin1 = lengths0*flowdir_HSMobject.cellsize*numpy.sin(theta)
                                #length0_cos1 = lengths0*flowdir_HSMobject.cellsize*numpy.cos(theta)
                                length0_sin2 = lengths0*out.cellsize*numpy.sin(-theta + pi/4)
                                length0_cos2 = lengths0*out.cellsize*numpy.cos(-theta + pi/4)
                                
                                length1_sin1 = lengths1*out.cellsize*numpy.sin(theta)
                                length1_cos1 = lengths1*out.cellsize*numpy.cos(theta)
                                #length1_sin2 = lengths1*flowdir_HSMobject.cellsize*numpy.sin(-theta + pi/4)
                                #length1_cos2 = lengths1*flowdir_HSMobject.cellsize*numpy.cos(-theta + pi/4)
    
                                in_np = (length1_sin1*length1_cos1) / (length1_sin1*length1_cos1 + length0_sin2*length0_cos2) + facet + 1
                                
                                del length0_sin2, length0_cos2, length1_sin1, length1_cos1, theta, facet
                                
                        elif mode == 'LINEARcos':
                            if out.aperture == 7:#this is for HHSM
                                facet, theta = divmod(in_np, pi/3)
                                length1 = numpy.cos(theta)*numpy.sin(theta)
                                length2 = numpy.cos(-theta + pi/3)*numpy.sin(-theta + pi/3)
                                in_np = length1/(length1 + length2) + facet + 1
                                del length1, length2, theta, facet
                            elif out.aperture == 9:#this is for RHSM
                                facet, theta = divmod(in_np, pi/4)
                                lengths0=facet%2
                                lengths1 = lengths0 * 2.0**0.5
                                lengths1[lengths1==0] = 1
                                lengths0[lengths0==0] = 2.0**0.5
                                in_np = lengths1*out.cellsize*numpy.sin(theta) / (lengths1*out.cellsize*numpy.sin(theta) + lengths0*out.cellsize*numpy.sin(-theta + pi/4)) + facet + 1
                        elif mode == 'CONTOUR':
                            #This code only works for concentrative cones
                            #Calculates ditace  across contour
                            #iterate
                            op = openHSM(in_file, self.HSM_object)
                            it =  numpy.nditer(in_np, flags=['multi_index'],op_flags=['readwrite'])
                            while not it.finished:
                                #find its IPord
                                cell = self.HSM_object._HHSM2HIP(in_array._v_pathname[len(str(level))+2:], in_array.name, it.multi_index)
                                #print cell
                                #determine the neighbourhood 
                                hood = op._Neighbourhood(cell[-min(self.HSM_object.level, self.HSM_object.level-level+1):], level)#Very ugly way to remove leading zero from path
                                #calculate the ratio
                                dir0 = hood[0]
                                if dir0 != self.HSM_object.NoDataValue:
                                    RHSM = dir0/(pi/4)+1
                                    flooring = int(math.floor(RHSM))%self.HSM_object.aperture
                                    ceiling = int(math.ceil(RHSM))%self.HSM_object.aperture
                                    contour = [hood[flooring], hood[ceiling]]
                                    ratio = (max(contour)-dir0)/(max(contour)-min(contour))+flooring
                                    if math.isnan(ratio) or ratio in [numpy.inf, -numpy.inf]:
                                        ratio = self.HSM_object.NoDataValue
                                    it[0]=ratio
                                it.iternext()
                            
                                
                           
                        in_np[pits] = 0
                        in_np[nodatas] = out.NoDataValue
                        #set to output data
                        out_array[:]=in_np
                        out_array.flush()
        #Skeptical about the below here. Particularly if used with linear
#        if self.HSM_object.aperture == 7:
#            flowdir_HSMobject._flipDinf()
            #for THSM this is not worth pursueing because of loops! However the THSM is:
            #direction = (faces[dir_index][0]-0.5*pi)/float(2*pi/(self.aperture-1)) + 1
                                      
        tii = time.time()
        print "Time to convert: {0}s".format(tii-ti)
        
        return out
            
            
            
#        else:

            

            
            #(B)or assign by linear ratio 
#            if self.aperture == 7:#this is for HHSM
#                theta = faces[dir_index][0] - (dir_index)*pi/3
#                linear = sin(theta) / (sin(theta) + sin(pi/3 - theta))
#                linear = sin(theta) / sin(pi/3)               
#            
#            elif self.aperture == 9:#this is for RHSM:
#                theta = faces[dir_index][0] - dir_index*pi/4
#                lengths = [self.cellsize, self.cellsize * 2.0**0.5]
#                linear = lengths[dir_index % 2]*sin(theta) / (lengths[dir_index % 2]*sin(theta) + lengths[(1+ dir_index) % 2]*sin(pi/4 - theta))
                
            #(C)or adjust for length
#            if self.aperture == 7:#this is for HHSM
#                #print [dir_index][0]
#                theta = faces[dir_index][0] - (dir_index+0.5)*pi/3
#                d1 = 3**0.5/2.0 * self.cellsize
#                d2 = tan(theta) * d1
#                d3 = d1/cos(theta)
#                linear = 0.5 + d2*float(d3)/float(self.cellsize)
#                #print theta
#                #print d1, d2, d3, linear
##                
#            elif self.aperture == 9:#this is for RHSM
#                theta = faces[dir_index][0] - dir_index*pi/4
#                lengths = [self.cellsize, self.cellsize * 2.0**0.5]
#                angles = [theta, pi/4 - theta]
#                ratio = [lengths[1]/(self.cellsize/float(cos(angles[dir_index % 2]))),(self.cellsize/float(cos(angles[dir_index % 2]))), lengths[1]]
#                linear = lengths[dir_index % 2]*sin(theta)/ (lengths[dir_index % 2]*sin(theta) + lengths[(1+ dir_index) % 2]*sin(pi/4 - theta)/ratio[dir_index % 2])
#                
#
#                
#            direction = linear + dir_index + 1#This line is required for (A) and (C)
#            #print direction
#
#        return direction
        
    def DinfDir(self, output, start_level = 0, stop_level = 1):
        #Basically the same as FlowDir but gives a d inf flow direction 
        #create the output HSM
        print "Calculating Dinf Directions"
        out = copy.deepcopy(self.HSM_object)
        out.h5 = output+'.h5'
        out.fname = output
        #delete h5 file it currently has the copied values.
        if os.path.exists(out.h5):
            os.remove(out.h5)
        out.CreateHIP()
        if stop_level > 1:
            out.PyrTree()#Creates blank files
            
        #open the input elevation data
        with tables.openFile(self.HSM_object.h5, mode = "r") as in_file:
            #open the output flow direction data
            with tables.openFile(out.h5, mode = "r+") as out_file:
                #create an openHSM object with the input elevation data
                op = openHSM(in_file, self.HSM_object)
                print "files open"
                
                    #go through the levels
                for level in range(start_level, stop_level):
                    base = "/L{0}".format(level)
                    #walk through the arrays on the specified level           
                    for in_array in in_file.walkNodes(base, "Array"):
                        #create an input numpy array
                        in_np = in_array.read()
                        #find its path name
                        out_array = out_file.getNode(in_array._v_pathname)
                        #get the output array with the same name
                        out_np = out_array.read()
                        #iterate the input elevation data
                        it =  numpy.nditer(in_np, flags=['multi_index'], op_flags=['readwrite'])
                        while not it.finished:
                            #find its IPord
                            cell = self.HSM_object._HHSM2HIP(in_array._v_pathname[len(str(level))+2:], in_array.name, it.multi_index)
                            #print cell
                            #determine the drop neighbourhood 
                            hood = op._Neighbourhood(cell[-min(self.HSM_object.level, self.HSM_object.level-level+1):], level)#Very ugly way to remove leading zero from path
                            #calculate the d inf direction
                            if numpy.equal(hood, self.HSM_object.NoDataValue).any():#first rule out no data
                                out_np[it.multi_index] = self.HSM_object.NoDataValue
                            else:
                                out_np[it.multi_index] = self.HSM_object._dinf2(hood)
                            it.iternext()
                            
                        #Update the output array (may be unnecassary)
                        out_array[:]=out_np
                        out_array.flush()                                               
        return out
        
    def DrawDirLines(self, FlowDirection, outputName, DirType = "DInf", start_level = 0, stop_level = 1, projection = None):         
        """creates an arrow interpretation of a flowdir
        call on direction HSM object
        based on lines
        DirType = "DInf"#Needs Dinf directions, currently draws lines at the fixed HHSM lengths rather than the direction dependent
        DirType = "Dx"#Needs HSNM directions
        Note: will append the level to the outputName, Output name can be the same as FlowDirection this will cause the output to appear in the same featuredataset"""
    
        arcpy.AddMessage('Drawing flow direction arrows')
        if self.HSM_object.aperture == 7:
            dir_HSM = HHSM(FlowDirection)
        if self.HSM_object.aperture == 9:
            dir_HSM = RHSM(FlowDirection)
        if self.HSM_object.aperture == 4:
            dir_HSM = THSM(FlowDirection)
        #in_flow_dir = dir_HSM.fname
        out_path = outputName    
        #Set environments
        arcpy.env.overwriteOutput = True
        arcpy.env.XYResolution = 0.0001
        arcpy.env.XYTolerance = 0.001
         
        #create the feature dataset 
        gdb, fname = os.path.split(outputName)
        spatial_ref = arcpy.SpatialReference()
        spatial_ref.loadFromString(projection)
        if not os.path.exists(gdb):
            arcpy.AddError("Draw levels failed\n" + gdb + " does not exist")
            return
            
        if not arcpy.Exists(gdb + fname):
            arcpy.CreateFeatureDataset_management (gdb, fname, spatial_ref)
            
        for level in range(start_level, stop_level):
            #create the output 
            out_name = r'{0}_lineDir_L{1}'.format(os.path.basename(outputName), level)
            template = 'Z:\Jo\Uni\PhD\models\Toolboxes\Templates.gdb\DirTemplate_Line'
            out_arrow = r'{0}\{1}'.format(out_path, out_name)
            print out_arrow
            arcpy.CreateFeatureclass_management(out_path, out_name, "POLYLINE", template, "DISABLED", "ENABLED", spatial_ref)
            base = '/L{0}'.format(level)
            
            #open output for editing
            with arcpy.da.InsertCursor(out_arrow, "SHAPE@") as cur:
                calc = calcHSM(dir_HSM)
                with tables.openFile(dir_HSM.h5) as in_file:
                    op = openHSM(in_file, dir_HSM) 
                    #iterate through the file                   
                    for in_array in in_file.walkNodes(base, "Array"):
                        array_IP = dir_HSM._HHSM2HIP(in_array._v_pathname, in_array.name, ())
                        if len(array_IP) >0:
                            arcpy.AddMessage('Processing array: {0}'.format(array_IP))
                        else:
                            arcpy.AddMessage('Processing array: (0,)')
                            
                        in_np = in_array.read()
                        it =  numpy.nditer(in_np, flags=['multi_index'], op_flags=['readwrite'])
                        while not it.finished:
                            #create and save the line geometry
                            if not int(it[0]) == dir_HSM.NoDataValue:
                                start_hip = array_IP + it.multi_index + (0,)*(level)
                                val_hip = (int(it[0]),) + (0,)*(level)
                                #print start_hip, val_hip, HSM_object.level
                                start = calc.IP2xy(start_hip)
                                if DirType == "Dx":
                                    end = calc.IP2xy(calc.IPadd(start_hip, val_hip, dir_HSM.level))
                                elif DirType == "DInf":
                                    dx = cos(it[0])*dir_HSM.cellsize*(7**0.5)**level
                                    dy = sin(it[0])*dir_HSM.cellsize*(7**0.5)**level
                                    end = (start[0]+dx, start[1]+dy)
                                geo = arcpy.Array([arcpy.Point(start[0], start[1]),
                                arcpy.Point(end[0], end[1])])
                                polyline = arcpy.Polyline(geo)
                                cur.insertRow([polyline])
            
                            it.iternext()
                    del op    
        
    def DinfAcc(self, FlowDirection, output, start_level = 0, stop_level = 1):
        #Recursive climbing flow accumulation tool. Capable of interpreting dinf directions. Note that it uses HSM directions not radians, if necessary apply Dinf2HSM first
        sys.setrecursionlimit(8000)#python defaults at 1000 but we need more
        #how high you can go is platform dependent
        #Misc\find_recursionlimit.py
         
        arcpy.AddMessage('Accumulating: {0}'.format(FlowDirection))
         #Create the output
        if not stop_level:
            stop_level = self.HSM_object.level        
        
        #create the output HHSM
        out_HSM = copy.deepcopy(self.HSM_object)
        out_HSM.h5 = output+'.h5'
        out_HSM.fname = output
        if os.path.exists(out_HSM.h5):
            os.remove(out_HSM.h5)
        #nodata = out_HSM.NoDataValue
        #out_HSM.NoDataValue = 0 #Use the Nodata value to set all values to zero to 0
        out_HSM.CreateHIP()
        #out_HSM.NoDataValue = nodata
        if stop_level > 1:
            out_HSM.PyrTree()#Creates blank files 
        #open the input direction data
        if self.HSM_object.aperture == 7:
            dir_HSM = HHSM(FlowDirection)
        if self.HSM_object.aperture == 9:
            dir_HSM = RHSM(FlowDirection)
        if self.HSM_object.aperture == 4:
            dir_HSM = THSM(FlowDirection)
         
        with tables.openFile(dir_HSM.h5, mode = "r") as in_dir_file:
            #open the output flow acc data
            with tables.openFile(out_HSM.h5, mode = "r+") as out_file:
                #create an openHSM object with the input direction data
                op_dir = openHSM(in_dir_file, dir_HSM)
                #create an openHSM object with the output accumulation data
                op_acc = openHSM(out_file, out_HSM)
                #create the acc object
                acc = AccObj(dir_HSM, out_HSM, op_dir, op_acc)                
                #go through the levels
                for level in range(start_level, stop_level):
                    #print 'level: {0}'.format(level)
                    base = "/L{0}".format(level)
                    #walk through the arrays on the specified level           
                    for out_array in out_file.walkNodes(base, "Array"):
                        #create an output numpy array
                        out_np = out_array.read()
                        #find its path name and get the direction array with the same name
#                        dir_array = in_dir_file.getNode(out_array._v_pathname)
#                        dir_np = dir_array.read()
                        #iterate the output data
                        it =  numpy.nditer(out_np, flags=['multi_index'], op_flags=['readwrite'])
                        while not it.finished:
                            #print it.multi_index
                            IP = tuple (int(val) for val in out_array.title[:-self.HSM_object.agg])+it.multi_index                      
                            acc._DPAREA(IP, level)
                            it.iternext()
        return out_HSM
        
    def FlowAcc_var(self,  FlowDirection, output):
        """
        determines dinf type flow accumulations on the direction sparse tree
        
        Parameters
        ----------
        FlowDirection: str
        Flow direction; uniform LoD flow direction array fname as text (must exist)
        output : str
        filename for the output flow acc HSM
        
        Output
        ------
        HSM object that contains links to the accumulation data in the sparse form
        branches L and E are empty
        
        Notes
        -----
        """
        sys.setrecursionlimit(8000)#python defaults at 1000 but we need more
        #how high you can go is platform dependent
        #Misc\find_recursionlimit.py
         
        arcpy.AddMessage('Accumulating Sparse: {0}'.format(FlowDirection))
        at0 = time.time()
         #Create the output     
        
        #create the output HHSM
        out_HSM = copy.deepcopy(self.HSM_object)
        out_HSM.h5 = output+'.h5'
        shutil.copyfile(FlowDirection+'.h5', out_HSM.h5)
        out_HSM.fname = output
        #Need preliminary sparse, set to 0
        with tables.openFile(out_HSM.h5, mode = "r+") as out_file:
            for node in out_file.iterNodes('/'):
                #print node._v_name
                if node._v_name != 'S':
                    out_file.removeNode("/", node._v_name, recursive = True)
            for node in out_file.walkNodes(where='/S', classname='Array'):
                node[0] = 0
                
                
#        if os.path.exists(out_HSM.h5):
#            os.remove(out_HSM.h5)
        #Need preliminary sparse, set to 0
        #nodata = out_HSM.NoDataValue
        #out_HSM.NoDataValue = 0 #Use the Nodata value to set all values to zero to 0
        #out_HSM.CreateHIP()
        #out_HSM.NoDataValue = nodata
        #open the input direction data
        if self.HSM_object.aperture == 7:
            dir_HSM = HHSM(FlowDirection)
        if self.HSM_object.aperture == 9:
            dir_HSM = RHSM(FlowDirection)
        if self.HSM_object.aperture == 4:
            dir_HSM = THSM(FlowDirection)
         
        with tables.openFile(dir_HSM.h5, mode = "r") as in_dir_file:
            #open the output flow acc data
            with tables.openFile(out_HSM.h5, mode = "r+") as out_file:
                #create an openHSM object with the input direction data
                op_dir = openHSM(in_dir_file, dir_HSM)
                #create an openHSM object with the output accumulation data
                op_acc = openHSM(out_file, out_HSM)
                #create the acc object
                acc = AccObj(dir_HSM, out_HSM, op_dir, op_acc) 
                #iterate the leaves
                for node in in_dir_file.walkNodes(where='/S', classname='Array'):
                    #convert the address into an IP index
                    IP = tuple(int(x) for x in node._v_pathname.split('/')[2:]) # node._v_name, node._v_depth
                    #print 'root: ' + str(IP)
                    acc._DPAREAvar(IP)
        
        at1 = time.time()            
        print "Time to accumulate spare: {0}".format(at1- at0)
        print
        return out_HSM

    def FlowAcc_RC(self, FlowDirection, output, start_level = 0, stop_level = 1):
        #Recursive climbing flow accumulation tool.
        #Adapted from array_DEMa.py
        #For uniform LoD arrays only
        #Inputs: Flow direction; uniform LoD flow direction array fname as text (must exist)
         #       output; string filename for the output flow acc HSM
         #       start_level; integer representing the finest level to accumulate (inclusive)
         #       stop_level; integer represnting the coarsest LoD to accumulate (exclusive)
         
         
        sys.setrecursionlimit(8000)#python defaults at 1000 but we need more
        #how high you can go is platform dependent
        #Misc\find_recursionlimit.py
         
        arcpy.AddMessage('Accumulating: {0}'.format(FlowDirection))
         #Create the output
        if not stop_level:
            stop_level = self.HSM_object.level        
        
        #create the output HHSM
        out_HSM = copy.deepcopy(self.HSM_object)
        out_HSM.h5 = output+'.h5'
        out_HSM.fname = output
        if os.path.exists(out_HSM.h5):
            os.remove(out_HSM.h5)
        #nodata = out_HSM.NoDataValue
        #out_HSM.NoDataValue = 0 #Use the Nodata value to set all values to zero to 0
        out_HSM.CreateHIP()
        #out_HSM.NoDataValue = nodata
        if stop_level > 1:
            out_HSM.PyrTree()#Creates blank files 
        #open the input direction data
        if self.HSM_object.aperture == 7:
            dir_HSM = HHSM(FlowDirection)
        if self.HSM_object.aperture == 9:
            dir_HSM = RHSM(FlowDirection)
        if self.HSM_object.aperture == 4:
            dir_HSM = THSM(FlowDirection)
         
        with tables.openFile(dir_HSM.h5, mode = "r") as in_dir_file:
            #open the output flow acc data
            with tables.openFile(out_HSM.h5, mode = "r+") as out_file:
                #create an openHSM object with the input direction data
                op_dir = openHSM(in_dir_file, dir_HSM)
                #create an openHSM object with the output accumulation data
                op_acc = openHSM(out_file, out_HSM)
                #create the acc object
                acc = AccObj(dir_HSM, out_HSM, op_dir, op_acc)                
                #go through the levels
                for level in range(start_level, stop_level):
                    print 'level: {0}'.format(level)
                    base = "/L{0}".format(level)
                    #walk through the arrays on the specified level           
                    for out_array in out_file.walkNodes(base, "Array"):
                        #create an output numpy array
                        out_np = out_array.read()
                        #find its path name and get the direction array with the same name
                        dir_array = in_dir_file.getNode(out_array._v_pathname)
                        dir_np = dir_array.read()
                        #iterate the output data
                        it =  numpy.nditer(out_np, flags=['multi_index'], op_flags=['readwrite'])
                        while not it.finished:
                            IP = tuple (int(val) for val in out_array.title[:-self.HSM_object.agg])+it.multi_index
                            if op_acc._get_vals(IP, level) == self.HSM_object.NoDataValue and dir_np[it.multi_index] != self.HSM_object.NoDataValue:
                                #create a dictionary include cell and give value 1
                                #print 'new stream: {0}'.format(IP)
                                acc.index[IP] = 1
                                #are there any upstream cells? 
                                acc._LookUp(IP, level)
                                # go downstream if possible
                                recipient = dir_np[it.multi_index]
                                #use while loop as only one or none out
                                #print 'recipient val : {0}'.format(recipient)
                                while recipient and recipient != self.HSM_object.NoDataValue:#if recipient is a direction reset IP accordingly
                                    IP_old = IP
                                    IP = acc.calc.IPadd(IP, (recipient,)) #locate the recipient
                                    if len(IP) < dir_HSM.level - level +1:
                                        #print 
                                        #print 'New Recip: {0}'.format(IP)
                                        acc.index[IP] = int(op_acc._get_vals(IP_old), level )+1#pass on the flow
                                        acc._LookUp(IP, level) #Look up
                                        recipient = int(op_dir._get_vals(IP, level)) #Continue down
                                    else:
                                        recipient = 0 #Where pointing off the dataset move on
                            it.iternext()
                    #assign catchment number to cells in SinkList
                    #Maybe only do this for non edge catchments.
#                    print sink,
#                    for item in SinkList:
#                        catchment[item[0],item[1]] = sink                  
#                    SinkIndex[sink] = [row,col] # and index the sink
#                    sink += 1
#                    SinkList[:]=[] #clear the list                        
            
        return out_HSM 
        
class AccObj(object):
    #everything you need for flowacc_RC
    def __init__(self, dir_object, acc_object, dir_op, acc_op):
        #self.HSM_object = HSM_object #input
        self.dir_object = dir_object #input
        self.acc_object = acc_object #output
        self.dir_op = dir_op #don't open these here so the open as can be used elsewhere. Ugly?
        self.acc_op = acc_op
        self.calc = calcHSM(dir_object)
        self.index = {}
        self.SinkList = []
        
        
    def _DPAREA(self, where, level = 0):
        if self.acc_op._get_vals(where, level) == self.acc_object.NoDataValue and self.dir_op._get_vals(where, level) != self.acc_object.NoDataValue:
            self.acc_op._set_vals(1, where, level)
            for i in range(1, self.dir_object.aperture):
                neighbour_IP = self.calc.IPadd(where, (i,))
                if len(neighbour_IP) < self.dir_object.level - level +1:#if inside the dataset
                    neighbour_dir = self.dir_op._get_vals(neighbour_IP, level)
                    if neighbour_dir not in (self.dir_object.NoDataValue, 0):
                        #determine the proportion of flow based on the difference between the neighbour index and the direction
                        r = abs(self.dir_object.IPnegation_table[i] - neighbour_dir)
                        #print where, i, neighbour_dir, r
                        if r > (self.dir_object.aperture - 2):#cope with closed arithmetic                        
                            r =  self.dir_object.aperture - 1 - r    
                        p = 1 - min(r, 1)#proportion
#                        print where, neighbour_IP
#                        print 'r: {0}, p: {1}'.format(r, p)
#                        print
                        if p > 0:
                            self._DPAREA(neighbour_IP, level)
                            acc = self.acc_op._get_vals(where, level) + p*self.acc_op._get_vals(neighbour_IP, level)
                            self.acc_op._set_vals(acc, where, level)
                    
                    
    def _DPAREAvar(self, where):
        #For use on variable neighbourhoods
        #get the dir neighbourhood

        COI = (0,1)
        #find neighbours
        hood = self.dir_op._VarNeighbourhood(where)
        #If accumulation cell is undefined and direction is defined
        if self.acc_op._get_vals(where, val_type = 'S')  == 0 and self.dir_op._get_vals(where, val_type = 'S') != self.acc_object.NoDataValue:
            #Assign the area of cell as initial accumulaton
            self.acc_op._set_vals(self.acc_object.aperture**(self.acc_object.level-len(where)), where, val_type = 'S')
            if where == COI:
                print hood 
            #Iterate the neighbourhood
            for hoodie in hood:#Need to skip centre?
                #all hoodies are inside the dataset
                (neighbour_IP, neighbour_dir) = hoodie
                #If neighbours direction is defined
                if neighbour_dir not in (self.dir_object.NoDataValue, 0, -1):
                    #If neighbour is a different LoD
                    if len(neighbour_IP) != len(where):
                        #Identify recipients (upper_IP and Lower_IP)
                        addend_lower = ((int(neighbour_dir) - 1)%(self.dir_object.aperture-1) + 1,)#(1-6 HHSM)
                        lower_IP = self.calc.IPadd(neighbour_IP, addend_lower, len(neighbour_IP))
                        addend_upper = ((int(math.ceil(neighbour_dir)) - 1)%(self.dir_object.aperture-1)  + 1,)#(1-6 HHSM)                       
                        upper_IP = self.calc.IPadd(neighbour_IP, addend_upper, len(neighbour_IP))
                        #From fine to coarse
                        if len(neighbour_IP) > len(where):
                            if where == COI:
                                print 'fine'
                                print where, neighbour_IP, neighbour_dir, addend_lower, lower_IP, addend_upper, upper_IP,
                            #finer neighbour
                            #If both within then 1
                            if lower_IP[:len(where)] == where and upper_IP[:len(where)] == where:
                                p = 1
                            #If niether within then 0
                            elif not lower_IP[:len(where)] == where and not upper_IP[:len(where)] == where:
                                p = 0
                            #If 1 within determine proportion
                            else:
                                if lower_IP[:len(where)] == where:
                                    r = abs(addend_lower[0] - neighbour_dir)
                                else:
                                    r = abs(addend_upper[0] - neighbour_dir)
                                #cope with closed arithmetic                         
                                if r > (self.dir_object.aperture - 2): 
                                    r =  self.dir_object.aperture - 1 - r    
                                p = 1 - min(r, 1)#proportion
                            if where == COI:
                                print 'prop: ' + str(p)
                        else:
                            #From coarse to fine
                            if where == COI:
                                print 'coarse'
                                print where, neighbour_IP, neighbour_dir, addend_lower, lower_IP, addend_upper, upper_IP,
                            if where[:len(lower_IP)] == lower_IP:
                                r = abs(addend_lower[0] - neighbour_dir)
                                #if lower_IP ==  upper_IP 
                            elif where[:len(upper_IP)] == upper_IP:
                                r = abs(addend_upper[0] - neighbour_dir)
                                
                            else:
                                r = 1
                                #cope with closed arithmetic                         
                            if r > (self.dir_object.aperture - 2): 
                                r =  self.dir_object.aperture - 1 - r    
                            p = 1 - min(r, 1)#proportion
                            if self.dir_object.aperture == 7:
                                p = p/2
                            elif self.dir_object.aperture == 9:
                                #(RHSM) Actually more complicated due to corner neighbours
                                if addend_upper[0] in [1,3,5,7]:
                                     p = p/3
                                     #else p=p
                            if where == COI:
                                print 'prop: ' + str(p)
                                
                    else:
                        #If neighbour is the same resolution
                        #need to define i the direction of neighbour + special cases
                        if where == COI:
                            print 'same'
                            print where, neighbour_IP, neighbour_dir,
                        neighbour_index = self.calc.IPadd(where, self.calc.IPnegation(neighbour_IP))
                        r = abs(neighbour_index[-1] - neighbour_dir)
                        #print where, i, neighbour_dir, r
                        if r > (self.dir_object.aperture - 2):#cope with closed arithmetic                        
                            r =  self.dir_object.aperture - 1 - r    
                        p = 1 - min(r, 1)#proportion
                        if where == COI:
                            print 'prop: ' + str(p)
                        #print where, neighbour_index, neighbour_IP, neighbour_dir, p
    #                        print where, neighbour_IP
    #                        print 'r: {0}, p: {1}'.format(r, p)
    #                        print

                    if p > 0:
                        #print "recursive: " + str(neighbour_IP)
                        self._DPAREAvar(neighbour_IP)
                        acc = self.acc_op._get_vals(where, val_type = 'S') + p*self.acc_op._get_vals(neighbour_IP, val_type = 'S')
                        if where == COI:
                            print 'where ' + str(where), str(self.acc_op._get_vals(where, val_type = 'S')), 'N_IP: ' + str(neighbour_IP),  str(self.acc_op._get_vals(neighbour_IP, val_type = 'S')), acc, where
                        self.acc_op._set_vals(acc, where, val_type = 'S')
                            
                            
    def _LookUp(self, where, level = 0):
        #For recursively traversy flow direction trees.
            for i in range(1, self.dir_object.aperture):#Check neighbours
                neighbour_IP = self.calc.IPadd(where, (i,))
                #print 'N IP: {0}'.format(neighbour_IP)
                #if len(neighbour_IP) < self.dir_object.level - level +1 or neighbour_IP[0] == 0:#if inside the dataset do I need to consider the leading 0 case?
                if len(neighbour_IP) < self.dir_object.level - level +1:#if inside the dataset
                    if int(self.dir_op._get_vals(neighbour_IP, level)) == self.dir_object.IPnegation_table[i] and int(self.acc_op._get_vals(neighbour_IP, level)) == self.acc_object.NoDataValue: #ie not a previously visited cell                        
                        self.index[neighbour_IP] = 0
                        for k, v in self.index.items():
                            self.index[k] = v + 1
                        self._LookUp(neighbour_IP, level)
            count = self.index.pop(where)
            self.acc_op._set_vals(count, where, level)#put accumulated flow in flow acc
            self.SinkList += [[where]] #keep track of catchment 
        
class NumPyRaster:
    #should move to seperate file as not related to hex_class
    
    def __init__(self, fname = False):
        #create a numpy array of values from an esri raster dataset and assign various properties as attributes
        #if no filename is provided default values will be used
        
        if fname: 
            #need to identify the appropriate nodata value
            raster_type = arcpy.GetRasterProperties_management(fname, "VALUETYPE")[0]
            arcpy.AddMessage('Esri raster type: {0}'.format(raster_type))
            esri_type_nodata = {'1':3, '2':15, '3':255, '4':-128, '5':65535, '6':-32768, '7':4294967295, '8':-2147483647, '9':-3.40282346639e+038}
            
            if raster_type in esri_type_nodata.keys():
                self.no_data_value = esri_type_nodata[raster_type] 
            else:
                arcpy.AddError("unknown nodata value") 
                self.no_data_value = -9999
                
               
            self.cellsizex = float(arcpy.GetRasterProperties_management(fname, "CELLSIZEX")[0])
            self.cellsizey = float(arcpy.GetRasterProperties_management(fname, "CELLSIZEy")[0])
            self.x0 = float(arcpy.GetRasterProperties_management(fname, "LEFT")[0])
            self.y0 = float(arcpy.GetRasterProperties_management(fname, "BOTTOM")[0])
            #data has been rotated 90 degrees clockwise (actually 270 degrees anticlockwise) so array can be indexed with cartesian coordinates
            self.data = numpy.rot90(arcpy.RasterToNumPyArray(fname, nodata_to_value = self.no_data_value), 3)

        else:
            self.cellsizex = 1
            self.cellsizey = 1
            self.x0 = 0
            self.y0 = 0
            self.data = numpy.array([[0,0],[0,0]])
            self.no_data_value = -9999
        
              
    def Export2Arc(self, filename, integer = False):
        arcpy.overwriteoutput = 1 #allows existing files to be over written
        #create a raster dataset out of the data array in the NumPyRaster object
        export_array = copy.deepcopy(numpy.rot90(self.data, 1)) #mysterious bug in arcpy.NumPyArrayToRaster regarding rotated arrays work around by passing a deep copy instead of the rotated array.
        output_raster = arcpy.NumPyArrayToRaster (export_array, arcpy.Point(self.x0,self.y0), self.cellsizex, self.cellsizey, value_to_nodata = self.no_data_value)
        
        if integer:
            output_raster = arcpy.sa.Int(output_raster)
        output_raster.save(filename)
        
        
    def AddBuffer(self):
        #adds a buffer of nodata cells to the NumPyRaster.data and updates the origin
        x, y = numpy.shape(self.data)
        AddSides = numpy.append(numpy.insert(self.data, 0, self.no_data_value, axis = 0), [[self.no_data_value]*y], axis = 0)
#
        self.data = numpy.append(numpy.insert(AddSides, 0, self.no_data_value, axis = 1), [[self.no_data_value]]*(x+2), axis = 1)
        del AddSides
        
        #need to update the origin to account for the buffer
        self.x0 -= self.cellsizex
        self.y0 -= self.cellsizey
    
        
    def StripBuffer(self):
        #removes the outer edge from the NumPyRaster.data regardless of the values in the edge
        #updates the origin
        self.data = self.data[1:-1, 1:-1]      
        self.x0 += self.cellsizex
        self.y0 += self.cellsizey
        
        
    def Attribute(self, other_raster):
        #sets the origin and cell size to match the other raster. Can be used in conjunction with a flow direction raster before calculate
        self.cellsizex = other_raster.cellsizex
        self.cellsizey = other_raster.cellsizey
        self.x0 = other_raster.x0
        self.y0 = other_raster.y0
        self.no_data_value = other_raster.no_data_value #may not be the appropriate value
        
    
    def StripNoDataEdges(self):
        #Removes edge cells if the entire edge (top, bottom, left, or right) contains no data.
        #generally wise to use early in the analysis (ie apply to DEM). Such cells may occur when clipping a DEM from a larger DEM using vector clipping polygons that do not align with cell edges. Flow acc and catchment results have edge effects if no data is not striped. If the edge no data cells are important they can be reapplied later using attributes.
        #The origin is adjusted to reflect the changes.
        x, y = numpy.shape(self.data)
        
        for i in range(x): 
            if numpy.any(self.data[i] != self.no_data_value):
                xi=i
                break
              
        for i in range(x-1,-1,-1): 
            if numpy.any(self.data[i] != self.no_data_value):
                xii=i+1
                break
        
        for i in range(y):
            if numpy.any(self.data[:,i] != self.no_data_value):
               yi = i
               break
              
        for i in range(y-1,-1,-1): 
            if numpy.any(self.data[:,i] != self.no_data_value):
                yii = i+1
                break


        self.data = self.data[xi:xii,yi:yii]
        
        self.x0 += self.cellsizex*xi
        self.y0 += self.cellsizey*yi

from __future__ import print_function

class Annotation(object):
    def __init__(self, **kwargs):
        self.name = kwargs.get('name', None)
        self.supercategory = kwargs.get('supercategory', None)
        self.area = kwargs.get('area', None)
        self.l = kwargs.get('l', None) # left
        self.t = kwargs.get('t', None) # top
        self.r = kwargs.get('r', None) # right
        self.b = kwargs.get('b', None) # bottom
        self.w = kwargs.get('w', None) # width
        self.h = kwargs.get('h', None) # height
        self.cat_id = kwargs.get('cat_id', None) # category_id
        self.id = kwargs.get('id', None)
        self.image_id = kwargs.get('image_id', None)
        self.is_crowd = kwargs.get('is_crowd', False)
        self.segmentation = kwargs.get('segmentation', None)

        if self.l == None or self.t == None:
            raise Exception("Annotation needs to be created with a left & a top (xmin, ymin)")
        if (self.r == None or self.b == None) and (self.w == None and self.h == None):
            raise Exception("Need xmax and ymax or width and height")
        if (self.r == None and self.b == None) and (self.w == None or self.h == None):
            raise Exception("Need xmax and ymax or width and height")

        if self.r:
            self.w = self.r - self.l
            self.h = self.b - self.t
        else:
            self.r = self.l + self.w
            self.b = self.t + self.h

        self.area = self.w * self.h

    def __str__(self):
        printout = "Name: {} \n".format(self.name)
        printout += "Category_id: {} \n".format(self.cat_id)
        printout += "bbox [x: {}, y: {}, w: {}, h: {}] \n".format(self.l, self.t, self.w, self.h)
        return printout

    def constrain_bbox(self, min_x, min_y, max_x, max_y):
        """
        If l, t, r, or b are not within min_x, min_y, max_x, or max_y,
        Force them to min/max.
        """
        if self.l < min_x:
            self.l = min_x
            #print("adjusted l")
        if self.t < min_y:
            self.t = min_y
            #print("adjusted t")
        if self.r > max_x:
            self.r = max_x
            #print("adjusted r")
        if self.b > max_y:
            self.b = max_y
            #print("adjusted b")

        self.w = self.r - self.l
        self.h = self.b - self.t

    def sanitize(self, catids_map=None, catid_to_name=None):
        """
        Force: - BBox values are all ints
               - width & height are ints and >= 1
               - area is updated
               - left & top are non-negative
               - is_crowd is bool
               - cat_id is int if it exists
               - name is all lowercase
               - find cat_id from catids map & name if exists
        """

        self.l = int(self.l)
        self.t = int(self.t)
        self.r = int(self.r)
        self.b = int(self.b)
        self.w = int(self.w)
        self.h = int(self.h)

        if self.area != self.w * self.h:
            self.area = self.w * self.h
        if self.l < 0:
            self.l = 0
        if self.t < 0:
            self.t = 0
        if self.h < 1:
            raise Exception("Height less than 0 on image_id {}".format(self.image_id))
        if self.w < 1:
            raise Exception("Width less than 0 on image_id {}".format(self.image_id))
        if self.cat_id != None:
            self.cat_id = int(self.cat_id)
        if self.is_crowd not in [True, False]:
            #print("WARNING: is_crowd not a bool, setting to False")
            self.is_crowd = False

        if self.name is not None:
            self.name = self.name.lower()
        # Find catid from name
        if catids_map is not None and self.name is not None:
            #print("{} : {}".format(self.name, catids_map.get(self.name, "sad")))
            self.cat_id = catids_map.get(self.name, self.cat_id)
            #print("WARNING: '{}' not in catids and not remapped".format(self.name))
        # Find name from catid
        if catid_to_name is not None and self.cat_id is not None:
            self.name = catid_to_name.get(self.cat_id, self.name)
            #print("WARNING: '{}' not in catids and not remapped".format(self.cat_id))

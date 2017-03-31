
class Range(object):
  """
  Class to represent range 
  """

  def __init__(self, start, end):
    self.start = start
    self.end = end
    self.window = (start, end)

class RangeTracker(object):
  """
  Class to manage, and track ranges
  """

  def __init__(self):
    #Using dictionary to make the search/delete operation faster for direct deletes.
    self.tracker = {}

  def add_range(self, r):
    self.tracker[r.window] = None


  def query_range(self, r):
    """
    This function converts the range to a dictionary of distinct values
    and then marks them as True, if it is part of any range. If all the elements
    are present in any of the ranges, True will be returned. Early return is used 
    for better performance
    """
    #Return True if the range is implicitly present
    if r.window in self.tracker:
      return True

    #Convert the range to a dict
    range_dict = self.get_range_dict(r)
 
    for ran in self.tracker:
      cur_range = Range(ran[0], ran[1])

      if self.is_overlapping(cur_range, r):
        for element in range(cur_range.start, cur_range.end + 1):
          try:
            range_dict[element] =  True
          except KeyError, k:
            pass
      if self.range_present(range_dict):
        return True

    return False

  def is_overlapping(self, r1, r2):
    """
    Validate if 2 ranges are overlapping or not
    """
    if r2.start > r1.end:
      return False
    elif r2.end < r1.start:
      return False

    return True

  def delete_range(self, r):
    """
    This function keeps track of all the modifications to the elements
    and updates the dictionary after the loop
    """
    #If the range is explicitly present, delete it
    if r.window in self.tracker:
      del self.tracker[r.window]
      return

    elements_to_be_added = []
    elements_to_be_deleted = []

    for ran in self.tracker:
      cur_range = Range(ran[0], ran[1])

      if self.is_overlapping(cur_range, r):

        elements_to_be_deleted.append(cur_range)
        if cur_range.start >= r.start and cur_range.end <= r.end:
          #If the range being deleted encompasses the current range, just skip the append part
          continue


        elif r.start >= cur_range.start and r.end <= cur_range.end:
          #If the range being deleted is contained within current range, rearrage the ranges
          if r.start  == cur_range.start:
            elements_to_be_added.append(Range(r.end + 1, cur_range.end))

          elif r.end == cur_range.end:
            elements_to_be_added.append(Range(cur_range.start, r.start - 1))

          else:
            elements_to_be_added.append(Range(cur_range.start, r.start -1))
            elements_to_be_added.append(Range(r.end + 1, cur_range.end))


        elif cur_range.start <= r.end:
          #All other cases adjust the existing ranges.
          elements_to_be_added.append(Range(r.end + 1, cur_range.end))

        elif cur_range.end >= r.start:
          elements_to_be_added.append(Range(cur_range.start, r.end -1))

    for e in elements_to_be_deleted:
      del self.tracker[e.window]
    for e in elements_to_be_added:
      self.add_range(e)
          
  def range_present(self, range_dict):
    """
    Utility method to find out if the range dict is full of 'True' values
    """
    for n in range_dict:
      if not range_dict[n]:
        return False
    return True

  def get_range_dict(self, ran):
    """
    Returns a dictioary with all the elements in a range with 'None' values
    i.e get_range_dict(1,3) will yeild {1: None, 2: None, 3: None}
    """
    d = {}
    for i in range(ran.start, ran.end + 1):
      d[i] = None

    return d
    

if __name__ == '__main__':
  rt = RangeTracker()

  rt.add_range(Range(10, 50))
  rt.add_range(Range(40, 80))

  assert rt.query_range(Range(1, 2)) == False
  assert rt.query_range(Range(30, 60)) == True
  assert rt.query_range(Range(10,50)) == True
  assert rt.query_range(Range(10,81)) == False

  rt.delete_range(Range(40, 50))

  assert (10,39) in rt.tracker
  assert (51,80) in rt.tracker

  rt.delete_range(Range(55, 59))

  assert rt.query_range(Range(55,58)) == False

  assert rt.query_range(Range(60,65)) == True

  assert rt.query_range(Range(90, 100)) == False

  assert (51,54) in rt.tracker
  assert (60,80) in rt.tracker

  rt.query_range(Range(75, 85))
  rt.delete_range(Range(60, 80))

  assert (60, 80) not in rt.tracker

  rt.delete_range(Range(0, 100))

  assert len(rt.tracker) == 0

  rt.add_range(Range(40, 80))
  rt.delete_range(Range(40, 80))



import time
def follow(thefile):
 	thefile.seek(0,2) # Go to the end of the file
 	while True:
 		line = thefile.readline()
 		if not line:
 			time.sleep(0.1) # Sleep briefly
 			continue
 		yield line


def grep(pattern,lines):
	for line in lines:
		if pattern in line:
			yield line

def grep2(pattern):
	print "Looking for %s" % pattern
	while True:
		line = (yield)
		if pattern in line:
 			print line,


if __name__ == '__main__':
	logfile = open("access-log")
	loglines = follow(logfile)
	pylines = grep("test",loglines)
	# Pull results out of the processing pipeline
	#for line in pylines:
	#	print line

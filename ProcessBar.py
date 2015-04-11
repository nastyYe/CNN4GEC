import sys
import time


#Output example: [===  ] 75%
#width define bar width
#percent define current percentage

def progress(width,percent,t=1):
	if t==1:
		print "%s %d%%\r" %(('%%-%ds' % width ) %(width*percent/100*'='),percent),
	else:
		print "%-2s%%\r" %percent,
	sys.stdout.flush()

	if percent >= 100:
		sys.stdout.flush()


if __name__ == "__main__":
	# this is for test!
	for i in xrange(100):
		progress(50,i+1,2)
		time.sleep(1)

	print "finish"

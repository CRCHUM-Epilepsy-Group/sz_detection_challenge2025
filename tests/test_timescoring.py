# Loading Annotations #


from timescoring.annotations import Annotation

# Annotation objects can be instantiated from a binary mask

fs = 1
mask = [0, 1, 1, 0, 0, 0, 1, 1, 1, 0]

labels = Annotation(mask, fs)

print('Annotation objects contain a representation as a mask and as a list of events:')
print(labels.mask)
print(labels.events)


# Annotation object can also be instantiated from a list of events
fs = 1
numSamples = 10  # In this case the duration of the recording in samples should be provided
events = [(1, 3), (6, 9)]

labels = Annotation(events, fs, numSamples)


# Computing performance score #

from timescoring import scoring
from timescoring import visualization

fs = 1
duration = 66 * 60
ref = Annotation([(8 * 60, 12 * 60), (30 * 60, 35 * 60), (48 * 60, 50 * 60)], fs, duration)
#hyp = Annotation([(8 * 60, 12 * 60), (28 * 60, 32 * 60), (50.5 * 60, 51 * 60), (60 * 60, 62 * 60)], fs, duration)
hyp = Annotation([(8 * 60, 12 * 60), (32 * 60, 51 * 60)], fs, duration)
scores = scoring.SampleScoring(ref, hyp)
figSamples = visualization.plotSampleScoring(ref, hyp)
print(ref)
print(hyp)
# Scores can also be computed per event
param = scoring.EventScoring.Parameters(
    toleranceStart=30,
    toleranceEnd=60,
    minOverlap=0,
    maxEventDuration=5 * 60,
    minDurationBetweenEvents=90)
scores = scoring.EventScoring(ref, hyp, param)
figEvents = visualization.plotEventScoring(ref, hyp, param)

print("# Event scoring\n" +
      "- Sensitivity : {:.2f} \n".format(scores.sensitivity) +
      "- Precision   : {:.2f} \n".format(scores.precision) +
      "- F1-score    : {:.2f} \n".format(scores.f1) +
      "- FP/24h      : {:.2f} \n".format(scores.fpRate))
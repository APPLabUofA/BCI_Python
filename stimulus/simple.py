import numpy as np
from pandas import DataFrame
from psychopy import visual, core, event
from time import time, strftime, gmtime
from optparse import OptionParser
import sys

#to call with LSL - python mac_run_exp.py --d 120 --s 1 --r 1 --e cueing
def present(duration=20,subject=1,session=1):
 
    n_trials = 10
    instruct = 0
    practicing = 0

    #seconds
    iti = 1
    iti_jitter = 0.2
    target_length = 0.05 

    record_duration = np.float32(duration)

    # Setup log
    cues = np.random.binomial(1, 0.5, n_trials)
    trials = DataFrame(dict(cues=cues))

    #Instructions function below
    if instruct:
        instructions()
    if practicing:
        practice()

    # graphics
    mywin = visual.Window([1440, 900], monitor="testMonitor", units="deg",
                          fullscr=True)

    mywin.mouseVisible = False

    fixation = visual.GratingStim(win=mywin, size=0.2, pos=[0, 0], sf=0)                            
    cuewin = visual.GratingStim(win=mywin, mask='circle', size=0.5, pos=[0, 1], sf=0)

    #Get ready screen
    text = visual.TextStim(
        win=mywin,
        text="Press right or up on each trial. Press space to begin",
        color=[-1, -1, -1],
        pos= [0,5])
    text.draw()
    fixation.draw()
    mywin.flip()
    event.waitKeys(keyList="space")
   

    #create a clock for rt's
    clock = core.Clock()
    #create a timer for the experiment and EEG markers
    start = time()

    for ii, trial in trials.iterrows():

        cue = trials['cues'].iloc[ii]

        # cue direction, pick target side
        if cue:
            cuewin.color = [1,0,0]
        else:
            cuewin.color = [0,0,1]
       
        ## Trial starts here ##
        # inter trial interval
        core.wait(iti + np.random.rand() * iti_jitter)

        # cueonset
        cuewin.draw()
        fixation.draw()
        t_cueOnset = time()
        mywin.flip()

        # response period
        core.wait(target_length)
        fixation.draw()
        t_respOnset = clock.getTime()
        mywin.flip()

        #Wait for response
        keys = event.waitKeys(keyList=["right","up"], timeStamped=clock)
        #categorize response
        correct = 1
        response = 1 
            
        if keys[0][0] == 'right':
            sys.stdout.write('\a')
        elif keys[0][0] == 'up':
            sys.stdout.write('\a')
    
        #reset sound        
        sys.stdout.flush()
 
         # block end
        if (time() - start) > record_duration:
            break
        event.clearEvents()

    #Goodbye Screen
    text = visual.TextStim(
        win=mywin,
        text="Thank you for participating. Press spacebar to exit the experiment.",
        color=[-1, -1, -1],
        pos= [0,5])
    text.draw()
    mywin.flip()
    event.waitKeys(keyList="space")

    mywin.mouseVisible = True

    # Cleanup
    mywin.close()



def practice():

   
    mywin.mouseVisible = False

   #Get ready screen
    text = visual.TextStim(
        win=mywin,
        text="Find the arrow keys, and begin fixating now. The first practice trial is about to begin",
        color=[-1, -1, -1],
        pos= [0,5])

    text.draw()
    fixation.draw()
    mywin.flip()
    core.wait(5)
   
    #End Practice Screen
    text = visual.TextStim(
        win=mywin,
        text="That is the end of the practice, Please let the experimenter know if you have any questions. Press Spacebar to begin the first trial.",
        color=[-1, -1, -1],
        pos= [0,5])
    
    text.draw()
    fixation.draw()
    mywin.flip()
    event.waitKeys(keyList="space")

    mywin.mouseVisible = True

    # Cleanup
    mywin.close()


def instructions():

    # graphics
    mywin = visual.Window([1440, 900], monitor="testMonitor", units="deg",
                          fullscr=True)

    mywin.mouseVisible = False

    #Instructions
    text = visual.TextStim(
        win=mywin,
        text="Welcome to the Attention Experiment!, Press spacebar to continue",
        color=[-1, -1, -1])
    text.draw()
    mywin.flip()
    event.waitKeys(keyList="space")

    mywin.mouseVisible = True

    mywin.close()


def main():
    present()

if __name__ == '__main__':
    main()
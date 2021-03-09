"""
http://www.swharden.com/blog/2013-05-09-realtime-fft-audio-visualization-with-python/
http://julip.co/2012/05/arduino-python-soundlight-spectrum/
"""

import ui_plot
import sys
import numpy as np
from PyQt5 import QtCore, QtWidgets
import qwt as Qwt
from recorder import *


try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

def plot_audio_and_detect_beats():
    if not input_recorder.has_new_audio: 
        return

    correlation, bpm = input_recorder.bpm()

    ac, tempo = input_recorder.bpm_librosa()
    tempo = np.mean(tempo) 
    input_recorder.has_new_audio = False

    uiplot.qwtPlot.setAxisScale(uiplot.qwtPlot.yLeft, 0, np.amax(correlation))
    # uiplot.qwtPlot.setAxisScale(uiplot.qwtPlot.yLeft, 0, np.amax(ac))

    # plot the data
    uiplot.btnC.setText(_fromUtf8("BPM Librosa: {:3.2f}".format(tempo)))
    uiplot.btnD.setText(_fromUtf8("BPM DWT: {:3.2f}".format(bpm)))
    c.setData(np.arange(len(correlation)),correlation)
    # c.setData(np.arange(len(ac)),ac)
    uiplot.qwtPlot.replot()
    
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    
    win_plot = ui_plot.QtWidgets.QMainWindow()
    uiplot = ui_plot.Ui_win_plot()
    uiplot.setupUi(win_plot)
    
    c = Qwt.QwtPlotCurve()  
    c.attach(uiplot.qwtPlot)
    
    uiplot.qwtPlot.setAxisScale(uiplot.qwtPlot.yLeft, 0, 100000)
    
    uiplot.timer = QtCore.QTimer()
    uiplot.timer.start(1.0)
    
    uiplot.timer.timeout.connect(plot_audio_and_detect_beats)
    
    input_recorder = InputRecorder()
    input_recorder.start()

    ### DISPLAY WINDOWS
    win_plot.show()
    code = app.exec_()

    # clean up
    input_recorder.close()
    sys.exit(code)
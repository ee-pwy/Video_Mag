import sys
sys.path.append('C:\Program Files\Vayyar\vtrigU\python')
import vtrigU

vtrigU.Init()

settings = vtrigU.RecordingSettings(vtrigU.FrequencyRange(65.0*1000,66.0*1000, 21), 30,
                                    vtrigU.VTRIG_U_TXMODE__LOW_RATE)

vtrigU.ApplySettings(settings)

vtrigU.Record()

recording = vtrigU.GetRecordingResult()
phasor_00 = recording[vtrigU.GetPairId(0,0)]
from .regimen import Regimen
from .plugins import Framerate, DetectNoProgress


r = Regimen()
r.use(Framerate())
r.use(DetectNoProgress(100))
r.use(ManualOverride)
r.use(OutputCsv)
r.use(FileMemory)
r.use(Render)
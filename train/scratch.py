from .regimen import Regimen
from .framerate import Framerate


r = Regimen()
r.use(Framerate)
r.use(DetectNoProgress)
r.use(ManualOverride)
r.use(OutputCsv)
r.use(FileMemory)
r.use(Render)
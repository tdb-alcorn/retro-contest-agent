from .regimen import Regimen


r = Regimen()
r.use(ManualOverride)
r.use(Framerate)
r.use(OutputCsv)
r.use(FileMemory)
r.use(Render)
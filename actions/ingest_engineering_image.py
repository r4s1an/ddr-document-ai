 
from dataclasses import dataclass
from pathlib import Path
from AI.gemini_pressure_plot import extract_pressure_plot
from services.plot_type_detector import detect_plot_type_bytes, PlotType
from services.save_pressure_plot import save_pressure_plot
from AI.gemini_pressure_profile import extract_pressure_profile
from services.save_pressure_profile import save_pressure_profile

@dataclass(frozen=True)
class EngineeringImageResult:
    plot_type: str
    plot_id: int

class IngestEngineeringImageAction:
    def __init__(self, engine, *, debug: bool = False):
        self.engine = engine
        self.debug = debug

    def execute(self, *, image_bytes: bytes, mime_type: str, source_key: str) -> EngineeringImageResult:

        det = detect_plot_type_bytes(image_bytes, debug=self.debug)

        with self.engine.begin() as conn:
            if det.plot_type == PlotType.PRESSURE_VS_TIME:

                extracted = extract_pressure_plot(image_bytes, mime_type)
                plot_id = save_pressure_plot(conn, source_key=source_key, extracted=extracted)

                return EngineeringImageResult(plot_type=det.plot_type.value, plot_id=plot_id)

            if det.plot_type == PlotType.PRESSURE_VS_DEPTH:

                extracted = extract_pressure_profile(image_bytes, mime_type)
                plot_id = save_pressure_profile(conn, source_key=source_key, extracted=extracted)
                return EngineeringImageResult(plot_type=det.plot_type.value, plot_id=plot_id)

            raise RuntimeError(f"Unhandled plot type: {det.plot_type}")

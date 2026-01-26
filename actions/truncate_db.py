from sqlalchemy import text

class TruncateDDRAction:
    def __init__(self, engine):
        self.engine = engine

    def execute(self):
        with self.engine.begin() as conn:
             
            conn.execute(text("TRUNCATE TABLE ddr_documents RESTART IDENTITY CASCADE;"))

class TruncatePressureTimePlotsAction:
    """
    Truncates all Pressure vs Time (Offset Wells) plot data.
    """
    def __init__(self, engine):
        self.engine = engine

    def execute(self):
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    TRUNCATE TABLE
                        pressure_plot
                    RESTART IDENTITY
                    CASCADE;
                    """
                )
            )

class TruncatePressureProfilePlotsAction:
    """
    Truncates all Pressure vs Depth (Pressure Profile) plot data.
    """
    def __init__(self, engine):
        self.engine = engine

    def execute(self):
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    TRUNCATE TABLE
                        pressure_profile_plot
                    RESTART IDENTITY
                    CASCADE;
                    """
                )
            )
class ModelConfig:
    # Default fallback model
    DEFAULT_MODEL = "glm-4-flash"
    
    # Agent specific models
    ROUTER_AGENT_MODEL = "glm-4-flash"
    TRIAGE_AGENT_MODEL = "glm-4-flash"
    APPOINTMENT_AGENT_MODEL = "glm-4-flash"
    INQUIRY_AGENT_MODEL = "glm-4-flash"
    REPORT_AGENT_MODEL = "glm-4-flash"
    
    # Specialized models
    VISION_MODEL = "glm-4v-plus"

    @classmethod
    def get_model(cls, agent_name: str) -> str:
        """Get the configured model for a specific agent name."""
        model_map = {
            "Router Agent": cls.ROUTER_AGENT_MODEL,
            "Triage Agent": cls.TRIAGE_AGENT_MODEL,
            "Appointment Agent": cls.APPOINTMENT_AGENT_MODEL,
            "Inquiry Agent": cls.INQUIRY_AGENT_MODEL,
            "Report Agent": cls.REPORT_AGENT_MODEL,
        }
        return model_map.get(agent_name, cls.DEFAULT_MODEL)


class ReActConfig:
    # Global enable/disable for ReAct
    ENABLE = True

    # Per-agent switches
    ROUTER_USE_REACT = False
    TRIAGE_USE_REACT = True
    APPOINTMENT_USE_REACT = True
    INQUIRY_USE_REACT = True
    REPORT_USE_REACT = True

    # Max thought-action iterations per query
    MAX_STEPS = 3

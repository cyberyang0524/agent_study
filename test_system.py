from agents.router_agent import RouterAgent
from agents.triage_agent import TriageAgent
from agents.appointment_agent import AppointmentAgent
from agents.inquiry_agent import InquiryAgent

def test_system():
    print("Testing Hospital Multi-Agent System...")
    
    router = RouterAgent()
    agents = {
        "triage": TriageAgent(),
        "appointment": AppointmentAgent(),
        "inquiry": InquiryAgent()
    }
    
    test_cases = [
        ("我头痛发烧，应该挂什么科？", "triage"),
        ("我想预约明天的王医生", "appointment"),
        ("医院在哪里？几点关门？", "inquiry"),
        ("肚子疼", "triage") # Should default to triage
    ]
    
    for user_input, expected_agent in test_cases:
        print(f"\nUser Input: {user_input}")
        
        # Test Router
        route_result = router.process(user_input)
        print(f"Router Output: {route_result}")
        
        if isinstance(route_result, dict) and "next_agent" in route_result:
            next_agent = route_result["next_agent"]
            if next_agent == expected_agent:
                print(f"PASS: Correctly routed to {next_agent}")
                
                # Test Agent Response
                agent = agents[next_agent]
                response = agent.process(user_input)
                print(f"Agent Response: {response}")
            else:
                print(f"FAIL: Expected {expected_agent}, got {next_agent}")
        else:
            print("FAIL: Router returned invalid format")

if __name__ == "__main__":
    test_system()

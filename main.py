import sys
from dotenv import load_dotenv

load_dotenv()

import colorama
from colorama import Fore, Style

from agents.router_agent import RouterAgent
from agents.triage_agent import TriageAgent
from agents.appointment_agent import AppointmentAgent
from agents.inquiry_agent import InquiryAgent
from agents.report_agent import ReportAnalysisAgent

def main():
    colorama.init()
    
    print(Fore.CYAN + "Welcome to the Hospital Customer Service Multi-Agent System!" + Style.RESET_ALL)
    print("Type 'exit' or 'quit' to end the conversation.\n")
    
    # Initialize agents
    router = RouterAgent()
    agents = {
        "triage": TriageAgent(),
        "appointment": AppointmentAgent(),
        "inquiry": InquiryAgent(),
        "report": ReportAnalysisAgent()
    }
    
    while True:
        try:
            user_input = input(Fore.GREEN + "You: " + Style.RESET_ALL).strip()
            if user_input.lower() in ["exit", "quit"]:
                print(Fore.CYAN + "System: Goodbye!" + Style.RESET_ALL)
                break
            
            if not user_input:
                continue
            
            # Check for image input simulation (e.g., "[image: url1, url2] text")
            images = []
            if user_input.startswith("[image:"):
                try:
                    end_bracket = user_input.find("]")
                    if end_bracket != -1:
                        image_part = user_input[7:end_bracket]
                        images = [url.strip() for url in image_part.split(",")]
                        user_input = user_input[end_bracket+1:].strip()
                        print(Fore.YELLOW + f"System: Detected {len(images)} image(s)." + Style.RESET_ALL)
                except Exception:
                    pass

            print(Fore.YELLOW + "System: Analyzing request..." + Style.RESET_ALL)
            
            # Step 1: Route the request
            route_result = router.process(user_input)
            
            # Check if router failed or returned error
            if isinstance(route_result, dict) and "next_agent" in route_result:
                next_agent_name = route_result["next_agent"]
                reason = route_result.get("reason", "No reason provided.")
                
                print(f"{Fore.MAGENTA}[Router]: Routing to {next_agent_name} agent. (Reason: {reason}){Style.RESET_ALL}")
                
                if next_agent_name in agents:
                    target_agent = agents[next_agent_name]
                    print(f"{Fore.BLUE}[{target_agent.name}]: ", end="", flush=True)
                    
                    # Enable streaming for agents
                    # Pass images if available
                    response_stream = target_agent.process(user_input, stream=True, images=images)
                    
                    # Handle stream
                    full_response = ""
                    try:
                        # Check if it's a generator/iterator
                        if hasattr(response_stream, '__iter__') and not isinstance(response_stream, (str, dict)):
                             for chunk in response_stream:
                                 # OpenAI/Zhipu chunk structure
                                 if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                                     delta = chunk.choices[0].delta
                                     if hasattr(delta, 'content') and delta.content is not None:
                                         content = delta.content
                                         print(content, end="", flush=True)
                                         full_response += content
                        else:
                            # Fallback if not streaming or error
                             print(response_stream, end="", flush=True)
                             full_response = str(response_stream)
                             
                        print(Style.RESET_ALL) # Newline after stream ends
                        
                        # Update agent memory
                        target_agent.update_memory("user", user_input)
                        target_agent.update_memory("assistant", full_response)
                        
                    except Exception as stream_err:
                        print(f"\nError streaming response: {stream_err}")

                else:
                    print(Fore.RED + f"Error: Agent '{next_agent_name}' not found." + Style.RESET_ALL)
            else:
                print(Fore.RED + "Error: Router failed to determine the next step." + Style.RESET_ALL)
                
        except KeyboardInterrupt:
            print("\n" + Fore.CYAN + "System: Goodbye!" + Style.RESET_ALL)
            break
        except Exception as e:
            print(Fore.RED + f"An error occurred: {e}" + Style.RESET_ALL)

if __name__ == "__main__":
    main()

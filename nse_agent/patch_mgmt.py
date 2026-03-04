"""patch_mgmt.py — Fix cfo_to_pat key name. Run: python patch_mgmt.py"""
f = open('agents/management_agent.py', encoding='utf-8').read()
f = f.replace('cfo_to_pat', 'cfo_to_net_income')
open('agents/management_agent.py', 'w', encoding='utf-8').write(f)
print('Fixed — now run: python -B main.py --ticker TCS --no-gpt --mode management')

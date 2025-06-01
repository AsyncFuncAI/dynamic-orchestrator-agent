# 🚀 DOA Framework - Beginner's Guide

## What is DOA? (30 seconds)

DOA learns to route requests between your services **automatically**. Instead of writing complex `if/else` logic, DOA figures out the best patterns by itself.

**Example**: You have a fast service and a slow service. DOA learns:
- Urgent requests → Fast service
- Detailed requests → Slow service
- All without you writing any routing code!

## Quick Start (2 minutes)

### Option 1: Copy & Paste Demo
```bash
# 1. Download this file
curl -O https://raw.githubusercontent.com/.../examples/copy_paste_start.py

# 2. Run it
python copy_paste_start.py
```

That's it! You'll see DOA learning routing patterns.

### Option 2: Try the Simple Demo
```bash
# 1. Clone the repo
git clone <repo-url>
cd doa-framework

# 2. Run the simple demo
python examples/simple_start.py
```

## What You'll See

```
🚀 DOA Framework - 2 Minute Demo

📋 BEFORE training (random routing):
Request 'urgent' → SlowService → TerminatorAgent
Request 'detailed' → FastService → TerminatorAgent

🧠 Training DOA...
✅ Training done!

📋 AFTER training (learned routing):
Request 'urgent' → FastService → TerminatorAgent  
Request 'detailed' → SlowService → TerminatorAgent

🎉 DOA learned:
• Urgent requests → FastService
• Detailed requests → SlowService
• No hardcoded if/else needed!
```

## Use With Your Code (5 minutes)

### Step 1: Wrap Your Functions
```python
from doa_framework import AgentInterface, AgentOutput

# Your existing function
def my_fast_function(data):
    return "quick result"

# Wrap it as DOA agent
class MyFastAgent(AgentInterface):
    def __init__(self):
        super().__init__("MyFastAgent")
    
    def execute(self, state):
        # Call your function here
        result = my_fast_function(state.task_specification)
        return AgentOutput(result, cost=0.5)
```

### Step 2: Create DOA System
```python
from doa_framework import *

# Your agents
agents = [MyFastAgent(), MySlowAgent(), TerminatorAgent()]

# DOA components (copy-paste this)
policy = PolicyNetwork(32, len(agents), 64)
orchestrator = Orchestrator(agents, policy, RewardConfig())
```

### Step 3: Use It
```python
def handle_request(request_data):
    state = SystemState(json.dumps(request_data), [], 0, 3)
    trajectory = orchestrator.run_episode(state)
    # DOA automatically picked the best agents!
    return extract_results(trajectory)
```

## Real Examples

### Web API Routing
```python
# Before: Hardcoded routing
if request.urgency == "high":
    return fast_service(request)
else:
    return slow_service(request)

# After: DOA learns the pattern
return doa.handle(request)
```

### Database vs Cache
```python
# Before: Manual logic
if cache.has(key):
    return cache.get(key)
else:
    result = database.get(key)
    cache.set(key, result)
    return result

# After: DOA learns cache-first pattern
return doa.handle({"type": "lookup", "key": key})
```

### Microservice Orchestration
```python
# Before: Complex orchestration code
def process_order(order):
    if inventory.check(order.items):
        payment = payment_service.charge(order)
        if payment.success:
            shipping.schedule(order)
            notifications.send(order.user)

# After: DOA learns the workflow
return doa.handle({"type": "order", "data": order})
```

## Why DOA vs Traditional Approaches?

| Traditional | DOA Framework |
|-------------|---------------|
| Write complex if/else logic | DOA learns patterns automatically |
| Hard to maintain routing rules | Self-improving system |
| Static decision making | Adaptive to changing conditions |
| Manual optimization | Automatic cost/quality optimization |

## Common Questions

### Q: Is this just for AI/ML apps?
**A**: No! DOA works with any services - databases, APIs, microservices, functions, etc.

### Q: Do I need to know machine learning?
**A**: No! Just wrap your functions and DOA handles the learning.

### Q: What if DOA makes wrong decisions?
**A**: DOA learns from feedback. Bad decisions get corrected over time.

### Q: How much setup is required?
**A**: Minimal! See the 2-minute copy-paste example above.

### Q: Can I use this in production?
**A**: Yes! Start with shadow mode, then gradually roll out.

## Next Steps

1. **Try the demo**: Run `copy_paste_start.py`
2. **Adapt to your code**: Replace the demo functions with yours
3. **See the magic**: Watch DOA learn your patterns
4. **Go deeper**: Check `examples/api_router_demo.py` for advanced usage

## Getting Help

- **Simple questions**: Check `examples/simple_start.py`
- **Advanced usage**: See `examples/integration_example.py`
- **Real demo**: Run `examples/api_router_demo.py`
- **Issues**: Open GitHub issue

## The Bottom Line

DOA replaces complex routing logic with a learning system. Instead of writing and maintaining complicated `if/else` statements, you let DOA figure out the best patterns automatically.

**Start simple, see the value, then expand!** 🚀

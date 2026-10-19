import asyncio
from mavsdk import System

async def run():
    drone = System()
    await drone.connect(system_address="udpin://0.0.0.0:14540")

    print("Dang cho ket noi voi drone...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Da ket noi!")
            break

    print("Dang cho cam bien san sang...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("San sang bay!")
            break

    print("Arm va cat canh")
    await drone.action.arm()
    await drone.action.takeoff()
    await asyncio.sleep(10)

    print("Ha canh")
    await drone.action.land()

asyncio.run(run())

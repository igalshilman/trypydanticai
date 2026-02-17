import restate

"""Example of Pydantic AI with multiple tools which the LLM needs to call in turn to answer a question.

In this case the idea is a "weather" agent — the user can ask for the weather in multiple cities,
the agent will use the `get_lat_lng` tool to get the latitude and longitude of the locations, then use
the `get_weather` tool to get the weather.

Run with:
   uv add uvicorn

   uv run uvicorn run main:app 
"""


import asyncio
from dataclasses import dataclass
from typing import Any

from httpx import AsyncClient
from pydantic import BaseModel

from pydantic_ai import Agent, RunContext


@dataclass
class Deps:
    client: AsyncClient


weather_agent = Agent(
    'openai:gpt-5-mini',
    # 'Be concise, reply with one sentence.' is enough for some models (like openai) to use
    # the below tools appropriately, but others like anthropic and gemini require a bit more direction.
    instructions='Be concise, reply with one sentence.',
    deps_type=Deps,
    retries=2,
)


class LatLng(BaseModel):
    lat: float
    lng: float


@weather_agent.tool
async def get_lat_lng(ctx: RunContext[Deps], location_description: str) -> LatLng:
    """Get the latitude and longitude of a location.

    Args:
        ctx: The context.
        location_description: A description of a location.
    """
    # NOTE: the response here will be random, and is not related to the location description.
    r = await ctx.deps.client.get(
        'https://demo-endpoints.pydantic.workers.dev/latlng',
        params={'location': location_description},
    )
    r.raise_for_status()
    return LatLng.model_validate_json(r.content)


@weather_agent.tool
async def get_weather(ctx: RunContext[Deps], lat: float, lng: float) -> dict[str, Any]:
    """Get the weather at a location.

    Args:
        ctx: The context.
        lat: Latitude of the location.
        lng: Longitude of the location.
    """
    # NOTE: the responses here will be random, and are not related to the lat and lng.
    temp_response, descr_response = await asyncio.gather(
        ctx.deps.client.get(
            'https://demo-endpoints.pydantic.workers.dev/number',
            params={'min': 10, 'max': 30},
        ),
        ctx.deps.client.get(
            'https://demo-endpoints.pydantic.workers.dev/weather',
            params={'lat': lat, 'lng': lng},
        ),
    )
    temp_response.raise_for_status()
    descr_response.raise_for_status()
    return {
        'temperature': f'{temp_response.text} °C',
        'description': descr_response.text,
    }
    

from restate.ext.pydantic import RestateAgent 
restate_weather_agent = RestateAgent(weather_agent)


weather = restate.Service("weather")


@weather.handler()
async def run(ctx: restate.Context, prompt: str) -> str:
    async with AsyncClient() as client:
        deps = Deps(client=client)
        result = await restate_weather_agent.run(
            prompt, deps=deps
        )
        return result.output


app = restate.app([weather])

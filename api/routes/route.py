import logging

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, HTTPException

from containers.app_container import AppContainer
from controller.llm_controller import LLMController
from data_model.api.response import (
    LLMInput,
    LLMOutput,
    MeasureSpeedInput,
    MeasureSpeedOutput
    )

router = APIRouter(
    prefix="/LLM",
    tags=["LLM"],
    responses={404: {"description": "Not found"}, 500: {"description": "server error"}}
)
logger = logging.getLogger("route")

@router.post(
    path="/generate_text",
    tags=["LLM"]
)
@inject
async def generate_text(
    data: LLMInput,
    llm_controller: LLMController = Depends(Provide[AppContainer.llm_controller]),
):
    """
    Generate text
    :param data: data input
    :param llm_controller: LLM controller
    :return:
    """

    prompt = data.prompt
    max_length = data.max_length

    output: LLMOutput = llm_controller.generate_text(
        prompt=prompt, 
        max_length=max_length
    )
    
    if not output.error:
        return output
    else:
        raise HTTPException(
            status_code=500,
            detail=output.error
        )
    
@router.post(
    path="/measure_speed",
    tags=["LLM"]
)
@inject
async def measure_speed(
    data: MeasureSpeedInput,
    llm_controller: LLMController = Depends(Provide[AppContainer.llm_controller]),
):
    """
    Generate text
    :param data: data input
    :param llm_controller: LLM controller
    :return:
    """

    prompt = data.prompt
    num_iterations = data.num_iterations

    output: MeasureSpeedOutput = llm_controller.measure_speed(
        prompt=prompt, 
        num_iterations=num_iterations
    )
    
    if not output.error:
        return output
    else:
        raise HTTPException(
            status_code=500,
            detail=output.error
        )

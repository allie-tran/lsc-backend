import httpx
from configs import SUBMIT_URL
from fastapi import APIRouter, HTTPException

from submit.models import (
    AnswerItem,
    AnswerSet,
    DRESSubmitRequest,
    DRESSubmitResponse,
    SubmitAnswerRequest,
    SubmitAnswerResponse,
)

submit_router = APIRouter()


@submit_router.post("/submit-answer", response_model=SubmitAnswerResponse)
async def submit_answer(request: SubmitAnswerRequest):
    """
    Submit an answer to DRES
    """
    if not request.session_id:
        raise HTTPException(status_code=401, detail="Please log in")

    async with httpx.AsyncClient() as client:
        if request.query_type == "QA":
            assert isinstance(request.answer, str), "Answer must be a string"
            answer = [AnswerItem(text=request.answer)]
        else:
            if isinstance(request.answer, str):
                answer = [
                    AnswerItem(
                        media_item_name=request.answer,
                    )
                ]
            else:
                answer = [
                    AnswerItem(
                        media_item_name=ans,
                    )
                    for ans in request.answer
                ]

        print(answer)

        dres_request = DRESSubmitRequest(
            answer_sets=[
                AnswerSet(
                    task_id="",
                    task_name="",
                    answers=answer,
                )
            ],
        )

        print(dres_request.model_dump(by_alias=True, exclude_unset=True))
        url = f"{SUBMIT_URL}/{request.evaluation_id}?session={request.session_id}"
        dres_response = await client.post(
            url, json=dres_request.model_dump(by_alias=True, exclude_unset=True)
        )

        try:
            dres_response.raise_for_status()
        except httpx.HTTPStatusError as e:
            return SubmitAnswerResponse(
                severity="error",
                message=f"Error submitting answer: {e}",
                verdict="ERROR",
            )

        response = DRESSubmitResponse.model_validate(dres_response.json())

        match (response.status, response.submission):
            case (False, _):
                return SubmitAnswerResponse(
                    severity="error",
                    message=response.description,
                    verdict="INVALID",
                )
            case (_, "INVALID"):
                return SubmitAnswerResponse(
                    severity="error",
                    message=response.description,
                    verdict="INVALID",
                )
            case (_, "ERROR"):
                return SubmitAnswerResponse(
                    severity="error",
                    message=response.description,
                    verdict="ERROR",
                )
            case (_, "CORRECT"):
                return SubmitAnswerResponse(
                    severity="success",
                    message=response.description,
                    verdict="CORRECT",
                )
            case (_, "INCORRECT"):
                return SubmitAnswerResponse(
                    severity="warning",
                    message=response.description,
                    verdict="INCORRECT",
                )
            case (_, "INDETERMINATE"):
                return SubmitAnswerResponse(
                    severity="info",
                    message=response.description,
                    verdict="INDETERMINATE",
                )
            case _:
                raise HTTPException(status_code=500, detail="Unknown response")

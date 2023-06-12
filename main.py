from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, validator, Field
from typing import Optional
from tiktoken import get_encoding, list_encoding_names

ENCODERS = {}
for name in list_encoding_names():
    ENCODERS[name] = get_encoding(name)


app = FastAPI()


class EncodeRequest(BaseModel):
    text: str
    encoder: Optional[str] = Field(
        "gpt2",
        description="The encoder to use. Available encoders: " +
        ', '.join(list_encoding_names()) + ". Default: gpt2",
    )

    @validator('encoder')
    def is_encoder_in_list(cls, v):
        if v not in list_encoding_names():
            raise ValueError('Allowed encoders: ' +
                             ', '.join(list_encoding_names()))
        return v


class EncodeResponse(BaseModel):
    number_of_tokens: int
    tokens: list[int]
    encoder: str


class DecodeRequest(BaseModel):
    tokens: list[int]
    decoder: Optional[str] = "gpt2"


class DecodeResponse(BaseModel):
    text: str
    decoder: Optional[str] = Field(
        "gpt2",
        description="The decoder to use. Available decoders: " +
        ', '.join(list_encoding_names()) + ". Default: gpt2",
    )

    @validator('decoder')
    def is_decoder_in_list(cls, v):
        if v not in list_encoding_names():
            raise ValueError('Allowed decoders: ' +
                             ', '.join(list_encoding_names()))
        return v


@app.get('/', include_in_schema=False, response_class=HTMLResponse)
async def root():
    return HTMLResponse(content="""
      <html>
        <head>
          <title>Tokens</title>
        </head>
        <body>
          API Documentation: <a href="/docs">/docs</a><br>
        </body>
      </html>
      """, status_code=200)


@app.post("/tokens/encode",
          response_model=EncodeResponse,
          status_code=200,
          tags=["tokens"],
          summary="Encodes the text and counts the number of tokens",
          description="Encodes the text and counts the number of tokens")
async def count_tokens(request: EncodeRequest):
    """ Encodes the text and counts the number of tokens """
    encoded_text = ENCODERS[request.encoder].encode(request.text)
    return EncodeResponse(number_of_tokens=len(encoded_text), tokens=encoded_text, encoder=request.encoder)


@app.post("/tokens/decode",
          response_model=DecodeResponse,
          status_code=200,
          tags=["tokens"],
          summary="Decodes a list of tokens and returns the text",
          description="Decodes a list of tokens and returns the text")
async def count_tokens(request: DecodeRequest):
    """ Decodes a list of tokens and returns the text """
    decoded = ENCODERS[request.decoder].decode(request.tokens)
    return DecodeResponse(text=decoded, decoder=request.decoder)

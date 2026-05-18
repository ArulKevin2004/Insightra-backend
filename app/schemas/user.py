from pydantic import BaseModel, EmailStr


class UserBase(BaseModel):
    username: str
    email: EmailStr
    is_active: bool = True


class User(UserBase):
    id: int


class UserCreate(UserBase):
    password: str


class UpdatePassword(BaseModel):
    current_password: str
    new_password: str


class ResetPassword(BaseModel):
    token: str
    new_password: str


import dartmouth_auth
from langchain_dartmouth.exceptions import InvalidKeyError

class AuthenticatedMixin:
    """A mix-in class to faciliate authentication"""

    def authenticate(self, jwt_url=None):
        try:
            if self.authenticator:
                jwt = self.authenticator()
            else:
                jwt = dartmouth_auth.get_jwt(
                    dartmouth_api_key=self.dartmouth_api_key, jwt_url=jwt_url
                )
        except KeyError as e:
            raise InvalidKeyError from e

        auth_header = {"Authorization": f"Bearer {jwt}"}
        self._set_extra_headers(auth_header)

    def _set_extra_headers(self, headers):
        if hasattr(self.client, "headers"):
            if self.client.headers is not None:
                self.client.headers.update(headers)
            else:
                self.client.headers = headers
        if hasattr(self.client, "_client"):
            if self.client._client._custom_headers is not None:
                self.client._client._custom_headers.update(headers)
            else:
                self.client._client._custom_headers = headers
        if hasattr(self, "async_client"):
            if hasattr(self.async_client, "headers"):
                if self.async_client.headers is not None:
                    self.async_client.headers.update(headers)
                else:
                    self.async_client.headers = headers
            if hasattr(self.async_client, "_client"):
                if self.async_client._client._custom_headers is not None:
                    self.async_client._client._custom_headers.update(headers)
                else:
                    self.async_client._client._custom_headers = headers

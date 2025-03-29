"""Modern dependency injection container using Protocol-based interfaces."""

import inspect
from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Optional,
    Type,
    TypeVar,
    get_type_hints,
    cast,
)

from .config.settings import settings

# Generic type variables for better type safety
T = TypeVar("T")
D = TypeVar("D")  # Type for dependencies


class ServiceContainer:
    """A fully type-based dependency injection container using Protocol interfaces."""

    def __init__(self):
        """Initialize an empty container."""
        self._factories: Dict[Type, Callable] = {}  # Type-based factories
        self._instances: Dict[Type, Any] = {}  # Type-based instances

        # Automatically register settings as a singleton
        self._instances[settings.__class__] = settings

    def register(
        self, service_type: Type[T]
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Register a service type using a decorator pattern.

        Args:
            service_type: The type/interface to register

        Returns:
            A decorator function that registers the factory

        Example:
            @container.register(DocumentRepository)
            def create_repository(settings: Settings) -> DocumentRepository:
                return SQLiteDocumentRepository(settings.database.path)
        """

        def decorator(factory: Callable[..., T]) -> Callable[..., T]:
            self._factories[service_type] = factory
            return factory

        return decorator

    def get(self, service_type: Type[T]) -> T:
        """Get a service instance by type.

        Args:
            service_type: Type of the service to retrieve

        Returns:
            The service instance

        Raises:
            KeyError: If the service is not registered
        """
        # Return existing instance if available
        if service_type in self._instances:
            return cast(T, self._instances[service_type])

        # Create new instance if we have a factory
        if service_type in self._factories:
            # Get the factory function
            factory = self._factories[service_type]

            # Auto-resolve dependencies based on type hints
            instance = self._create_instance_with_dependencies(factory)
            self._instances[service_type] = instance
            return cast(T, instance)

        raise KeyError(
            f"Service of type {service_type.__name__} not registered in container"
        )

    def _create_instance_with_dependencies(self, factory: Callable[..., T]) -> T:
        """Create a service instance with auto-resolved dependencies."""
        # Get type hints and signature
        hints = get_type_hints(factory)
        signature = inspect.signature(factory)
        kwargs: Dict[str, Any] = {}

        # Process each parameter
        for param_name, param in signature.parameters.items():
            # Skip parameters with default values
            if param.default is not inspect.Parameter.empty:
                continue

            # Skip return type annotation
            if param_name == "return":
                continue

            # Get parameter type from type hints
            if param_name in hints:
                param_type = hints[param_name]

                # Handle Optional types
                if (
                    hasattr(param_type, "__origin__")
                    and param_type.__origin__ is Optional
                ):
                    try:
                        # Get the first type argument (T in Optional[T])
                        real_type = param_type.__args__[0]
                        kwargs[param_name] = self.get(real_type)
                    except KeyError:
                        # If not found, provide None for Optional parameters
                        kwargs[param_name] = None
                else:
                    # Recursively resolve regular dependencies
                    try:
                        kwargs[param_name] = self.get(param_type)
                    except KeyError as e:
                        raise KeyError(
                            f"Could not resolve dependency {param_name}: {e}"
                        ) from e

        # Create the instance with dependencies injected
        return factory(**kwargs)

    def __getitem__(self, service_type: Type[T]) -> T:
        """Enable container[ServiceType] syntax for type-safe access.

        Args:
            service_type: Type of service to retrieve

        Returns:
            The service instance cast to the appropriate type

        Example:
            repo = container[DocumentRepository]  # Type-safe access
        """
        return cast(T, self.get(service_type))

    def has(self, service_type: Type) -> bool:
        """Check if a service is registered by type.

        Args:
            service_type: Type to check

        Returns:
            True if the service is registered, False otherwise
        """
        return service_type in self._instances or service_type in self._factories

    def singleton(
        self, service_type: Type[T]
    ) -> Callable[[Callable[..., T]], Callable[[], T]]:
        """Register a singleton service that is created only once.

        Args:
            service_type: The type/interface to register

        Returns:
            A decorator function that registers the singleton

        Example:
            @container.singleton(DatabaseConnection)
            def create_database() -> DatabaseConnection:
                return DatabaseConnection(settings.database.url)
        """

        def decorator(factory: Callable[..., T]) -> Callable[[], T]:
            @wraps(factory)
            def singleton_factory() -> T:
                if service_type not in self._instances:
                    instance = self._create_instance_with_dependencies(factory)
                    self._instances[service_type] = instance
                return cast(T, self._instances[service_type])

            self._factories[service_type] = singleton_factory
            return singleton_factory

        return decorator

    def clear(self):
        """Clear all cached instances but keep factory registrations."""
        self._instances.clear()
        # Re-register settings as a singleton
        self._instances[settings.__class__] = settings


# Create the global container instance
container = ServiceContainer()


# This function sets up the container with all service registrations
def setup_container():
    """Set up the container with all service registrations.

    This function is idempotent - it can be called multiple times safely.
    Services that have already been initialized will be reused.

    This implementation uses fully type-based dependency injection with Protocol classes.
    """
    # Skip setup if container is already initialized
    from ..domain.repositories.document_repository import DocumentRepository

    if container.has(DocumentRepository):
        return

    # Import the services module to register type-based services
    # This will automatically register all services with the container
    from . import services

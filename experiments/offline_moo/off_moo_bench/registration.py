import re

from offline_moo.off_moo_bench.task import Task

# the format of allowed task names using regex
TASK_PATTERN = re.compile(r"(\w+)-(\w+)-v(\d+)$")


# error message for when an improper name is used
MISMATCH_MESSAGE = (
    "Attempted to register malformed task name: {}. ("
    "Currently all names must conform to regex template {}.)"
)


# error message when a task version is not found but other versions are
DEPRECATED_MESSAGE = "Task {} not found (versions include {})"


# error message when an problem is not registered with a dataset
problem_MESSAGE = "problem {} not found with Dataset {} (problems include {})"


# error message when no tasks are found with a particular name
UNKNOWN_MESSAGE = "No registered task with name: {}"


# error message when a task with this name is already specified
REREGISTRATION_MESSAGE = "Cannot re-register id: {}"


class TaskSpecification(object):

    def __init__(
        self, task_name, dataset, problem, dataset_kwargs=None, problem_kwargs=None
    ):
        """Create a specification for a model-based optimization task that
        dynamically imports that task when self.make is called.

        Arguments:

        task_name: str
            the name of the model-based optimization task which must match
            with the regex expression given by TASK_PATTERN
        dataset: str or callable
            the import path to the target dataset class or a callable that
            returns the target dataset class when called
        problem: str or callable
            the import path to the target problem class or a callable that
            returns the target problem class when called
        dataset_kwargs: dict
            additional keyword arguments that are provided to the dataset
            class when it is initialized for the first time
        problem_kwargs: dict
            additional keyword arguments that are provided to the problem
            class when it is initialized for the first time

        """

        # store the task name and task arguments
        self.task_name = task_name
        self.dataset = dataset
        self.problem = problem

        # instantiate default keyword arguments
        self.dataset_kwargs = dataset_kwargs if dataset_kwargs else {}
        self.problem_kwargs = problem_kwargs if problem_kwargs else {}

        # check if the name matches with the regex template
        match = TASK_PATTERN.search(task_name)

        # if there is no match raise a ValueError
        if not match:
            raise ValueError(MISMATCH_MESSAGE.format(task_name, TASK_PATTERN.pattern))

        # otherwise select the dataset and problem name with regex
        self.dataset_name, self.problem_name = match.group(1), match.group(2)

    def make(self, dataset_kwargs=None, problem_kwargs=None, **kwargs):
        """Instantiates the intended task using the additional
        keyword arguments provided in this function.

        Arguments:

        dataset_kwargs: dict
            additional keyword arguments that are provided to the dataset
            class when it is initialized for the first time
        problem_kwargs: dict
            additional keyword arguments that are provided to the problem
            class when it is initialized for the first time

        Returns:

        task: Task
            an instantiation of the task specified by task name which
            conforms to the standard task api

        """

        # use additional_kwargs to override self.kwargs
        dataset_kwargs_final = self.dataset_kwargs.copy()
        if dataset_kwargs is not None:
            dataset_kwargs_final.update(dataset_kwargs)

        # use additional_kwargs to override self.kwargs
        problem_kwargs_final = self.problem_kwargs.copy()
        if problem_kwargs is not None:
            problem_kwargs_final.update(problem_kwargs)

        # return a task composing this problem and dataset
        return Task(
            self.dataset,
            self.problem,
            dataset_kwargs=dataset_kwargs_final,
            problem_kwargs=problem_kwargs_final,
            **kwargs
        )

    def __repr__(self):
        return (
            "TaskSpecification({}, {}, {}, "
            "dataset_kwargs={}, problem_kwargs={})".format(
                self.task_name,
                self.dataset,
                self.problem,
                self.dataset_kwargs,
                self.problem_kwargs,
            )
        )


class TaskRegistry(object):

    def __init__(self):
        """Provide a global interface for registering model-based
        optimization tasks that remain stable over time

        """

        self.task_specs = {}

    def make(self, task_name, dataset_kwargs=None, problem_kwargs=None, **kwargs):
        """Instantiates the intended task using the additional
        keyword arguments provided in this function.

        Args:

        task_name: str
            the name of the model-based optimization task which must match
            with the regex expression given by TASK_PATTERN
        dataset_kwargs: dict
            additional keyword arguments that are provided to the dataset
            class when it is initialized for the first time
        problem_kwargs: dict
            additional keyword arguments that are provided to the problem
            class when it is initialized for the first time

        Returns:

        task: Task
            an instantiation of the task specified by task name which
            conforms to the standard task api

        """

        return self.spec(task_name).make(
            dataset_kwargs=dataset_kwargs, problem_kwargs=problem_kwargs, **kwargs
        )

    def all(self):
        """Generate a list of the names of all currently registered
        model-based optimization tasks

        Returns:

        names: list
            a list of names that corresponds to all currently registered
            tasks where names[i] is suitable for self.make(names[i])

        """

        return self.task_specs.values()

    def spec(self, task_name):
        """Looks up the task specification identifed by the task
        name argument provided here

        Args:

        task_name: str
            the name of the model-based optimization task which must match
            with the regex expression given by TASK_PATTERN

        Returns:

        spec: TaskSpecification
            a specification whose make function will dynamically import
            and create the task specified by 'name'

        """

        # check if the name matches with the regex template
        match = TASK_PATTERN.search(task_name)

        # if there is no match raise a ValueError
        if not match:
            raise ValueError(MISMATCH_MESSAGE.format(task_name, TASK_PATTERN.pattern))

        # try to locate the task specification
        try:
            return self.task_specs[task_name]

        # if it does not exist try to find out why
        except KeyError:

            # make a list of all similar registered tasks
            dataset_name, problem_name = match.group(1), match.group(2)

            matching = [
                valid_name
                for valid_name, valid_spec in self.task_specs.items()
                if dataset_name == valid_spec.dataset_name
                and problem_name == valid_spec.problem_name
            ]

            # there is another version available
            if matching:
                raise ValueError(DEPRECATED_MESSAGE.format(task_name, matching))

            matching = [
                valid_name
                for valid_name, valid_spec in self.task_specs.items()
                if dataset_name == valid_spec.dataset_name
            ]

            # there is another problem available
            if matching:
                raise ValueError(
                    problem_MESSAGE.format(problem_name, dataset_name, matching)
                )

            # there are no similar matching tasks
            else:
                raise ValueError(UNKNOWN_MESSAGE.format(task_name))

    def register(
        self, task_name, dataset, problem, dataset_kwargs=None, problem_kwargs=None
    ):
        """Register a specification for a model-based optimization task that
        dynamically imports that task when self.make is called.

        Args:

        task_name: str
            the name of the model-based optimization task which must match
            with the regex expression given by TASK_PATTERN
        dataset: str or callable
            the import path to the target dataset class or a callable that
            returns the target dataset class when called
        problem: str or callable
            the import path to the target problem class or a callable that
            returns the target problem class when called
        dataset_kwargs: dict
            additional keyword arguments that are provided to the dataset
            class when it is initialized for the first time
        problem_kwargs: dict
            additional keyword arguments that are provided to the problem
            class when it is initialized for the first time

        """

        # raise an error if that task is already registered
        if task_name in self.task_specs:
            raise ValueError(REREGISTRATION_MESSAGE.format(task_name))

        # otherwise add that task to the collection
        self.task_specs[task_name] = TaskSpecification(
            task_name,
            dataset,
            problem,
            dataset_kwargs=dataset_kwargs,
            problem_kwargs=problem_kwargs,
        )


# create a global task registry
registry = TaskRegistry()


def register(task_name, dataset, problem, dataset_kwargs=None, problem_kwargs=None):
    """Register a specification for a model-based optimization task that
    dynamically imports that task when self.make is called.

    Args:

    task_name: str
        the name of the model-based optimization task which must match
        with the regex expression given by TASK_PATTERN
    dataset: str or callable
        the import path to the target dataset class or a callable that
        returns the target dataset class when called
    problem: str or callable
        the import path to the target problem class or a callable that
        returns the target problem class when called
    dataset_kwargs: dict
        additional keyword arguments that are provided to the dataset
        class when it is initialized for the first time
    problem_kwargs: dict
        additional keyword arguments that are provided to the problem
        class when it is initialized for the first time

    """

    return registry.register(
        task_name,
        dataset,
        problem,
        dataset_kwargs=dataset_kwargs,
        problem_kwargs=problem_kwargs,
    )


def make(task_name, dataset_kwargs=None, problem_kwargs=None, **kwargs):
    """Instantiates the intended task using the additional
    keyword arguments provided in this function.

    Args:

    task_name: str
        the name of the model-based optimization task which must match
        with the regex expression given by TASK_PATTERN
    dataset_kwargs: dict
        additional keyword arguments that are provided to the dataset
        class when it is initialized for the first time
    problem_kwargs: dict
        additional keyword arguments that are provided to the problem
        class when it is initialized for the first time

    Returns:

    task: Task
        an instantiation of the task specified by task name which
        conforms to the standard task api

    """

    return registry.make(
        task_name,
        dataset_kwargs=dataset_kwargs,
        problem_kwargs=problem_kwargs,
        **kwargs
    )


def spec(task_name):
    """Looks up the task specification identifed by the task
    name argument provided here

    Args:

    task_name: str
        the name of the model-based optimization task which must match
        with the regex expression given by TASK_PATTERN

    Returns:

    spec: TaskSpecification
        a specification whose make function will dynamically import
        and create the task specified by 'name'

    """

    return registry.spec(task_name)

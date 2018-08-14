from gym.envs.registration import register
import logging

LOGGER = logging.getLogger(__name__)

_REGISTERED = False


def register_custom_envs():
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    LOGGER.info("Registering multiworld pygame gym environments")
    register(
        id='Point2DLargeEnv-v0',
        entry_point='multiworld.envs.pygame.point2d:Point2DEnv',
        tag={
            'git-commit-hash': '166f0f3',
            'author': 'Vitchyr'
        },
        kwargs={
            'target_radius': 1,
            'ball_radius': 1,
        },
    )


register_custom_envs()

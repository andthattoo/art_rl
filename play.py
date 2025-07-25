import asyncio
import art
from art.rewards import ruler_score_group
from openai.types.chat.chat_completion import Choice
from openai.types.chat import ChatCompletionMessage


async def main():
    # Initial messages shared by all trajectories
    initial_messages = [
        {
            "role": "system",
            "content": "You are a comedy writer. Generate funny jokes based on the given topic.",
        },
        {"role": "user", "content": "Tell me a funny joke about computers"},
    ]

    # Create three trajectories with different quality responses
    good_trajectory = art.Trajectory(
        messages_and_choices=[
            *initial_messages,
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content="Why don't computers ever get invited to parties?\n\nBecause they always crash! 🥁\n\nBut seriously, have you tried turning them off and on again?",
                ),
            ),
        ],
        reward=0.0,
    )

    mediocre_trajectory = art.Trajectory(
        messages_and_choices=[
            *initial_messages,
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content="What do you call a computer that doesn't work?\n\nBroken.",
                ),
            ),
        ],
        reward=0.0,
    )

    off_topic_trajectory = art.Trajectory(
        messages_and_choices=[
            *initial_messages,
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content="I don't really know jokes about computers, but here's a fact: The sky is blue because of Rayleigh scattering.",
                ),
            ),
        ],
        reward=0.0,
    )

    # Create a TrajectoryGroup and use RULER to score
    group = art.TrajectoryGroup(
        [good_trajectory, mediocre_trajectory, off_topic_trajectory]
    )
    judged_group = await ruler_score_group(group, "openai/o3", debug=True)

    # Display rankings
    if judged_group:
        sorted_trajectories = sorted(
            judged_group.trajectories, key=lambda t: t.reward, reverse=True
        )
        for rank, traj in enumerate(sorted_trajectories, 1):
            messages = traj.messages()
            print(f"Rank {rank}: Score {traj.reward:.3f}")
            print(f"  Response: {messages[-1]['content'][:50]}...")


asyncio.run(main())

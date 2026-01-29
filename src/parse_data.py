"""Parser for Weibo‑style cascade datasets.

The expected input format is tab‑separated with the following columns:

    message_id<TAB>root_user<TAB>publish_time<TAB>retweet_number<TAB>retweet_paths

`retweet_paths` is a space‑separated list of entries of the form
`u1/u2/…/un:Δt` where `u1` is the root of the diffusion, `un` is the user
who retweeted, and `Δt` is the elapsed time in seconds since the publish
time when that retweet occurred.  Intermediate user IDs reflect the
path through which the message was propagated.

Only the earliest retweet by each user is kept.  The parser emits
warnings if the declared `retweet_number` does not match the number of
unique retweets discovered.
"""

from __future__ import annotations

import logging
from typing import List

from .cascade import Cascade, Event, earliest_event_by_user


def parse_dataset(path: str) -> List[Cascade]:
    """Parse a dataset file into a list of `Cascade` objects.

    Parameters
    ----------
    path: str
        Path to the input file.  Each line is parsed independently.

    Returns
    -------
    list of `Cascade`
        The parsed cascades.  Cascades with no events are ignored.
    """
    cascades: List[Cascade] = []
    with open(path, 'r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue  # skip blank lines
            parts = line.split('\t')
            if len(parts) < 5:
                logging.warning(f"Line {lineno}: expected at least 5 fields, got {len(parts)}")
                continue
            try:
                cid = int(parts[0])
                root = int(parts[1])
                publish_time = float(parts[2])
                retweet_number = int(parts[3])
            except ValueError as e:
                logging.warning(f"Line {lineno}: failed to parse numeric fields: {e}")
                continue
            cascade = Cascade(cid=cid, root=root, publish_time=publish_time)
            # parse retweet paths
            events: List[Event] = []
            retweets = parts[4].split(' ')
            for item in retweets:
                if not item:
                    continue
                try:
                    path_part, time_part = item.split(':')
                    time_rel = float(time_part)
                except ValueError:
                    logging.warning(f"Line {lineno}: malformed retweet entry '{item}'")
                    continue
                # path_part can have multiple segments separated by '/'
                users = path_part.split('/')
                if not users:
                    continue
                # The last user is the retweeter, the previous is the parent (if any)
                try:
                    user_ids = [int(u) for u in users]
                except ValueError:
                    logging.warning(f"Line {lineno}: non‑integer user ID in '{item}'")
                    continue
                if len(user_ids) == 1:
                    # direct retweet from root
                    child = user_ids[0]
                    parent = root
                else:
                    child = user_ids[-1]
                    parent = user_ids[-2]
                events.append(Event(user=child, parent=parent, time=time_rel))
            # deduplicate events by earliest occurrence
            deduped = earliest_event_by_user(events)
            if len(deduped) < retweet_number:
                logging.info(
                    f"Line {lineno}: declared {retweet_number} retweets but only {len(deduped)} unique users; duplicates were removed"
                )
            cascade.events = deduped
            cascade.sort_events()
            if cascade.n_events() == 0:
                continue
            cascades.append(cascade)
    return cascades

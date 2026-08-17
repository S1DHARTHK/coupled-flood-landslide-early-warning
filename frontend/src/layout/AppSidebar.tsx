import { useCallback } from "react";
import { Link, useLocation } from "react-router";

import { AlertIcon, BoltIcon, HorizontaLDots, ListIcon, PieChartIcon } from "../icons";
import { useSidebar } from "../context/SidebarContext";

type NavItem = {
  name: string;
  icon: React.ReactNode;
  path: string;
};

/**
 * Project navigation only. The four pages of the Kerala Flood–Landslide
 * Early Warning System — no template/demo entries.
 */
const navItems: NavItem[] = [
  { icon: <BoltIcon />, name: "Early Warning", path: "/early-warning" },
  { icon: <AlertIcon />, name: "Warning History", path: "/warnings" },
  { icon: <PieChartIcon />, name: "Analysis", path: "/analysis" },
  { icon: <ListIcon />, name: "Model Performance", path: "/model-performance" },
];

const AppSidebar: React.FC = () => {
  const { isExpanded, isMobileOpen, isHovered, setIsHovered } = useSidebar();
  const location = useLocation();

  const isActive = useCallback(
    (path: string) => location.pathname === path,
    [location.pathname]
  );

  const showLabels = isExpanded || isHovered || isMobileOpen;

  return (
    <aside
      className={`fixed mt-16 flex flex-col lg:mt-0 top-0 px-5 left-0 bg-white dark:bg-gray-900 dark:border-gray-800 text-gray-900 h-screen transition-all duration-300 ease-in-out z-50 border-r border-gray-200
        ${isExpanded || isMobileOpen ? "w-[290px]" : isHovered ? "w-[290px]" : "w-[90px]"}
        ${isMobileOpen ? "translate-x-0" : "-translate-x-full"}
        lg:translate-x-0`}
      onMouseEnter={() => !isExpanded && setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Project branding */}
      <div
        className={`py-6 flex ${
          !isExpanded && !isHovered ? "lg:justify-center" : "justify-start"
        }`}
      >
        <Link to="/early-warning" className="flex items-center gap-2.5">
          <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-brand-500 text-white">
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
              aria-hidden="true"
            >
              <path
                d="M12 3.5 21 19H3l9-15.5Z"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinejoin="round"
              />
              <path
                d="M12 9.5v4.2"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinecap="round"
              />
              <circle cx="12" cy="16.4" r="1" fill="currentColor" />
            </svg>
          </span>
          {showLabels && (
            <span className="flex flex-col leading-tight">
              <span className="text-sm font-semibold text-gray-800 dark:text-white/90">
                Kerala Early Warning
              </span>
              <span className="text-[10px] text-gray-500 dark:text-gray-400">
                Flood–Landslide System
              </span>
            </span>
          )}
        </Link>
      </div>

      {/* Navigation */}
      <div className="flex flex-col overflow-y-auto duration-300 ease-linear no-scrollbar">
        <nav className="mb-6">
          <h2
            className={`mb-4 text-xs uppercase flex leading-[20px] text-gray-400 ${
              !isExpanded && !isHovered ? "lg:justify-center" : "justify-start"
            }`}
          >
            {showLabels ? "Menu" : <HorizontaLDots className="size-6" />}
          </h2>

          <ul className="flex flex-col gap-2">
            {navItems.map((nav) => (
              <li key={nav.name}>
                <Link
                  to={nav.path}
                  className={`menu-item group ${
                    isActive(nav.path) ? "menu-item-active" : "menu-item-inactive"
                  }`}
                >
                  <span
                    className={`menu-item-icon-size ${
                      isActive(nav.path)
                        ? "menu-item-icon-active"
                        : "menu-item-icon-inactive"
                    }`}
                  >
                    {nav.icon}
                  </span>
                  {showLabels && <span className="menu-item-text">{nav.name}</span>}
                </Link>
              </li>
            ))}
          </ul>
        </nav>
      </div>
    </aside>
  );
};

export default AppSidebar;
